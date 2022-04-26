from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import torch, os, tqdm

def select_logprobs(logits, decoded_tokens, eos_id):
    logprobs = torch.nn.functional.log_softmax(logits, dim=2)

    selected_logprobs = []
    for i, generated_tokenized in enumerate(decoded_tokens):
        if eos_id in generated_tokenized:
            generated_tokenized = generated_tokenized[:generated_tokenized.index(eos_id)]
        selected_logprob = logprobs[i, torch.arange(len(generated_tokenized)), generated_tokenized]
        summed_logprob = torch.sum(selected_logprob)
        selected_logprobs.append(summed_logprob)
    selected_logprobs = torch.stack(selected_logprobs, dim=0)
    return selected_logprobs

models_folder = os.environ["MODELS_FOLDER"]

class GeneratorHF:
    def __init__(self, model_card="gpt2-medium", device="cuda", starter_file=None, gradient_checkpointing=False, max_enc_length=None, max_dec_length=None, force_dec_prepend=None):
        self.model_card = model_card

        self.is_gpt2 = "gpt2" in self.model_card or "summary_loop" in self.model_card or "keep_it_simple" in self.model_card
        if self.is_gpt2:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_card)
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_card)
        self.model.to(device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_card)
        self.gradient_checkpointing = gradient_checkpointing
        self.max_enc_length = max_enc_length
        self.max_dec_length = max_dec_length
        self.force_dec_prepend = force_dec_prepend

        if self.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.eval()

        if "facebook/wmt19" in self.model_card:
            self.tokenizer.pad_token = "<pad>"
            self.tokenizer.eos_token = "</s>"

        self.start_id = self.tokenizer.bos_token_id
        self.end_id = self.tokenizer.eos_token_id

        if "prophetnet" in self.model_card:
            # bos_token_id=102, eos_token_id=102
            self.start_id = 102
            self.end_id = 102

        if self.start_id is None and self.end_id is not None:
            # For MixQG
            self.start_id = 0

        self.device = device
        if self.is_gpt2:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        if starter_file is not None:
            self.reload(starter_file, strict=False)

    def reload(self, from_file, strict=True):
        if not os.path.isfile(from_file):
            # Try to look at the models folder for the file
            from_file = os.path.join(models_folder, from_file)
            assert os.path.isfile(from_file), "Starter file not found, in absolute or in models folder"

        loaded_dict = torch.load(from_file)
        print(self.model.load_state_dict(loaded_dict, strict=strict))

    def save(self, to_file):
        torch.save(self.model.state_dict(), to_file)

    def preprocess(self, encoded_texts, decoded_texts, max_enc_length=None, max_dec_length=None):

        assert len(encoded_texts) == len(decoded_texts), "Mismatch in input/output sizes"

        # encoder_tokenized = [torch.LongTensor(self.tokenizer.encode(text=text)) for text in encoded_texts]
        # encoder_ids = torch.nn.utils.rnn.pad_sequence(encoder_tokenized, batch_first=True, padding_value=0, truncation=True).to(self.device)

        encoder_ids = self.tokenizer.batch_encode_plus(encoded_texts, add_special_tokens=True, return_tensors="pt", padding=True, truncation=True).input_ids.to(self.device)

        if self.force_dec_prepend is not None:
            decoded_texts = [self.force_dec_prepend + text for text in decoded_texts]
        decoder_tokenized = [self.tokenizer.encode(text=text, add_special_tokens=False) for text in decoded_texts]

        decoder_ids_input = torch.nn.utils.rnn.pad_sequence([torch.LongTensor([self.start_id] + dec) for dec in decoder_tokenized], batch_first=True, padding_value=self.end_id).to(self.device)
        decoder_ids_output = torch.nn.utils.rnn.pad_sequence([torch.LongTensor(dec + [self.end_id]) for dec in decoder_tokenized], batch_first=True, padding_value=-1).to(self.device)

        if self.max_enc_length is not None and max_enc_length is None:
            max_enc_length = self.max_enc_length
        if self.max_dec_length is not None and max_dec_length is None:
            max_dec_length = self.max_dec_length

        if max_enc_length is not None:
            encoder_ids = encoder_ids[:, :max_enc_length]

        if max_dec_length is not None:
            decoder_ids_input = decoder_ids_input[:, :max_dec_length]
            decoder_ids_output = decoder_ids_output[:, :max_dec_length]

        return encoder_ids, decoder_ids_input, decoder_ids_output

    def train_batch(self, encoded_texts, decoded_texts, max_enc_length=None, max_dec_length=None, no_preinput=False):
        self.model.train()
        N = len(encoded_texts)

        encoder_ids, decoder_ids_input, decoder_ids_output = self.preprocess(encoded_texts, decoded_texts, max_enc_length, max_dec_length)

        crit = torch.nn.CrossEntropyLoss(ignore_index=-1)
        if self.is_gpt2:
            past = None
            if not no_preinput:
                encoder_output = self.model(input_ids=encoder_ids, past_key_values=None, return_dict=True, use_cache=True)
                past = encoder_output["past_key_values"]
            decoder_output = self.model(input_ids=decoder_ids_input, past_key_values=past, return_dict=True, use_cache=not self.gradient_checkpointing)
            logits = decoder_output["logits"]
        else:
            if no_preinput:
                encoder_ids = torch.LongTensor([[self.start_id]]).repeat(N, 1).to(self.device)
            model_output = self.model(input_ids=encoder_ids, decoder_input_ids=decoder_ids_input, return_dict=True, use_cache=not self.gradient_checkpointing)
            logits = model_output["logits"]

        N_unwrap = decoder_ids_output.shape[0] * decoder_ids_output.shape[1]
        loss = crit(logits.view(N_unwrap, -1), decoder_ids_output.contiguous().view(-1)) # self.tokenizer.vocab_size
        return loss

    def score_batch(self, encoded_texts, decoded_texts, max_enc_length=None, max_dec_length=None):
        encoder_ids, decoder_ids_input, decoder_ids_output = self.preprocess(encoded_texts, decoded_texts, max_enc_length, max_dec_length)

        with torch.no_grad():

            crit = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction="none")
            if self.is_gpt2:
                encoder_output = self.model(input_ids=encoder_ids, past_key_values=None, return_dict=True)
                past = encoder_output["past_key_values"]
                decoder_output = self.model(input_ids=decoder_ids_input, past_key_values=past, return_dict=True)
                logits = decoder_output["logits"]
            else:
                model_output = self.model(input_ids=encoder_ids, decoder_input_ids=decoder_ids_input, return_dict=True)
                logits = model_output["logits"]

            N, seqlength, vocab_size = logits.shape

            loss_components = crit(logits.view(N*seqlength, vocab_size), decoder_ids_output.contiguous().view(-1)).reshape(N, seqlength)
            num_words = torch.sum(decoder_ids_output != -1, dim=1)
            score_per_item = (- torch.sum(loss_components, dim=1) / num_words).tolist()
        return {"scores": score_per_item}

    def score(self, encoded_texts, decoded_texts, max_enc_length=None, max_dec_length=None, batch_size=32, progress=False):
        N = len(encoded_texts)
        iterator = range(0, N, batch_size)
        if progress and len(iterator) > 1:
            iterator = tqdm.tqdm(iterator)
        scores = []
        for i in iterator:
            batch_encoded_texts = encoded_texts[i:i+batch_size]
            batch_decoded_texts = decoded_texts[i:i+batch_size]
            batch_scores = self.score_batch(batch_encoded_texts, batch_decoded_texts, max_enc_length, max_dec_length)["scores"]
            scores += batch_scores
        return {"scores": scores}

    def generate(self, texts, max_enc_length=None, max_gen_length=None, num_runs=1, compute_logprobs=False, force_start=None, **gen_params):
        assert type(texts) == list, "The generate function takes as input a list of `str`"
        if len(texts) == 0:
            return []

        tokenized_paragraphs = [torch.LongTensor(self.tokenizer.encode(text=text)) for text in texts]
        tokenized_paragraphs = [tok_text for tok_text in tokenized_paragraphs for _ in range(num_runs)]

        decoder_input_ids = None
        if force_start is not None:
            decoder_input_ids = self.tokenizer.encode(force_start, return_tensors="pt", add_special_tokens=False)

        # Generate without leaving gradients
        with torch.no_grad():
            encoder_ids = torch.nn.utils.rnn.pad_sequence(tokenized_paragraphs, batch_first=True, padding_value=0).to(self.device)
            if max_enc_length is not None:
                encoder_ids = encoder_ids[:, :(max_enc_length-1)]
            N = encoder_ids.shape[0]
            start_column = torch.LongTensor([[self.start_id]] * N).to(self.device)
            encoder_ids = torch.cat((encoder_ids, start_column), dim=1)

            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.repeat(N, 1).to(self.device)
                if self.is_gpt2:
                    encoder_ids = torch.cat((encoder_ids, decoder_input_ids), dim=1)
                else:
                    decoder_input_ids = torch.cat((start_column, decoder_input_ids), dim=1)
                    gen_params["decoder_input_ids"] = decoder_input_ids

            _, input_seq_length = encoder_ids.shape
            if max_gen_length is not None:
                if self.is_gpt2:
                    gen_params["max_length"] = input_seq_length + max_gen_length
                else:
                    gen_params["max_length"] = max_gen_length

            if "num_beams" in gen_params: # Propagate param
                gen_params["num_return_sequences"] = gen_params["num_beams"]

            output_generate = self.model.generate(encoder_ids, return_dict_in_generate=True, output_scores=True, **gen_params)

        generated_ids = output_generate.sequences
        if self.is_gpt2 and decoder_input_ids is not None:
            generated_ids = torch.cat((decoder_input_ids, generated_ids), dim=1)
        if self.is_gpt2:
            generated_ids = generated_ids[:, input_seq_length:]

        N, gen_length = generated_ids.shape
        batch_size = len(texts)
        num_beams = N // (batch_size * num_runs)
        if num_beams > 1:
            # For some reason, they do not return a score if it is not beam-search...
            sequences_scores = output_generate.sequences_scores
        else:
            sequences_scores = torch.zeros(N).to(self.device)

        # The next block is to obtain logprobs... unfortunately have to run the model again, as there's no good book-keeping for HF beam-search
        selected_logprobs = torch.zeros(N).to(self.device)
        if compute_logprobs:
            # Don't run this unless we really need these (for RL training)
            expanded_encoder_ids = torch.repeat_interleave(encoder_ids, repeats=num_beams, dim=0)

            if self.is_gpt2:
                generated_input = torch.cat((torch.LongTensor([[self.start_id]] * N).to(self.device), generated_ids), dim=1)
                generated_output = torch.cat((generated_ids, torch.LongTensor([[self.end_id]] * N).to(self.device)), dim=1) # There is an error here, the end_id could be AFTER padding... need to fix

                expanded_encoder_ids = expanded_encoder_ids[:, :-1]

                encoder_output = self.model(input_ids=expanded_encoder_ids[:, :-1], past_key_values=None, return_dict=True)
                decoder_output = self.model(input_ids=generated_input, past_key_values=encoder_output.past_key_values, return_dict=True)

                selected_logprobs = utils_rl.select_logprobs(decoder_output.logits, generated_output.tolist(), self.end_id)
            else:
                expanded_encoder_ids = torch.repeat_interleave(encoder_ids, repeats=num_beams, dim=0)

                generated_input = generated_ids[:, :-1]
                generated_output = generated_ids[:, 1:]

                model_output = self.model(input_ids=expanded_encoder_ids, decoder_input_ids=generated_input, return_dict=True)
                selected_logprobs = utils_rl.select_logprobs(model_output.logits, generated_output.tolist(), eos_id=self.end_id)
            # print("Selected logprobs:", selected_logprobs.tolist())

        # Un-tokenize
        generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # Time to un-flatten
        num_candidates = num_runs * num_beams

        generated_texts = [generated_texts[i:(i+num_candidates)] for i in range(0, N, num_candidates)]
        selected_logprobs = [selected_logprobs[i:(i+num_candidates)] for i in range(0, N, num_candidates)]
        sequences_scores = [sequences_scores[i:(i+num_candidates)] for i in range(0, N, num_candidates)]

        outputs = []
        sort_by_key = "logprob" if compute_logprobs else "score"

        for gen_texts, scores, logprobs in zip(generated_texts, sequences_scores, selected_logprobs):
            output = [{"output_text": gen_text, "logprob": logprob, "score": score} for gen_text, score, logprob in zip(gen_texts, scores, logprobs)]
            output = sorted(output, key=lambda x: x[sort_by_key], reverse=True)
            outputs.append(output)

        return outputs


if __name__ == "__main__":
    # qgen = GeneratorHF(model_card="gpt2-medium", starter_file="/export/home/models/qgen/gpt2_med_newsqa_only_logprob_2.059.bin")
    # qgen = GeneratorHF(model_card="Salesforce/mixqg-large", starter_file="mixqgl_clean_qg_L_1.457.bin")
    qgen = GeneratorHF(model_card="facebook/bart-large", starter_file="/export/home/models/bartl_clean_qg_L_1.917.bin")
    paragraph = "Liu Qiangdong, also known as Richard Liu, CEO of JD.com, raises his arms to celebrate the IPO for his company at the Nasdaq MarketSite, New York, May 22, 2014."

    for start in ["Why", "How", "What"]:
        print(qgen.generate([paragraph], force_start=start, max_gen_length=20)[0][0]["output_text"])
    exit()

    # gpt2zs = GeneratorHF(model_card="gpt2-large")
    # document = "US President Joe Biden spoke at a news conference Thursday at the NATO headquarters in Brussels, Belgium, after meeting with other world leaders of NATO, the European Council and the G7. The key global figures are seeking to align their responses to Russia's invasion of Ukraine. The President touched upon the unity of NATO, the prospect of Russian President Vladimir Putin using chemical weapons, and the possible role of China in the conflict. Biden took questions from reporters and spoke for roughly 30 minutes. TL;DR:"
    # print(gpt2zs.generate([document], num_runs=1, max_gen_length=100))

    # exit()
    paragraphs = ["On Tuesday, the Joint Committee on Administrative Rules (JCAR) voted against extending the Illinois Department of Public Health (IDPH) emergency rule on school mask mandates."]
    gen2 = GeneratorHF(model_card="gpt2-medium", starter_file="qgen/gpt2_med_newsqab_sched_logprob_1.793.bin")
    # gen2.eval()

    batch_outs2 = gen2.generate(paragraphs, max_gen_length=20, do_sample=True, num_runs=3)
    for outs2 in batch_outs2:
        print("=========")
        for out2 in outs2:
            print("[%.3f] %s" % (out2["logprob"], out2["output_text"]))
        print("--------")


    exit()
    gen = GeneratorHF(model_card="philippelaban/keep_it_simple")
    # paragraph = """A small capsule containing asteroid soil samples that was dropped from 136,700 miles in space by Japan's Hayabusa2 spacecraft landed as planned in the Australian Outback on December 6. The extremely high precision required to carry out the mission thrilled many in Japan, who said they took pride in its success."""
    paragraph = """Earth travels a tremendous distance in its orbit around the sun, at a speed of around 30km/s or over 108000km per hour."""
    outs = gen.generate([paragraph], max_length=150, num_beams=4, do_sample=True, num_return_sequences=4)[0]
    for out in outs:
        print("[%.3f] %s" % (out["score"], out["output_text"]))
        print()
    # gens = [out["output_text"] for out in outs]
    # inps = [paragraph] * len(gens)

    inps = ["Earth travels a tremendous distance in its orbit around the sun, at a speed of around 30km/s or over 108000km per hour."] * 2
    gens = ["Earth travels a tremendous size in its orbit around the sun, at a speed of around 30 km/s or over 108000km.", "The experiment The Earth travels very quickly -LRB- 100,000 km per hour -RRB- around the Sun ."]

    print(gen.score(inps, gens))

    # from model_generator import Generator
    import utils_misc, utils_squad
    utils_misc.select_freer_gpu()

    paragraph = "The Palazzo Pitti (Italian pronunciation: [paˈlattso ˈpitti]), in English sometimes called the Pitti Palace, is a vast, mainly Renaissance, palace in Florence, Italy. It is situated on the south side of the River Arno, a short distance from the Ponte Vecchio. The core of the present palazzo dates from 1458 and was originally the town residence of Luca Pitti an ambitious Florentine banker."

    answer = "Luca Pitti"

    marked_paragraph = utils_squad.mark_paragraph_answer(paragraph, answer, model_card="Salesforce/mixqg-large")
    print(">>>", marked_paragraph)

    gen = GeneratorHF(model_card="Salesforce/mixqg-large")

    gen_out = gen.generate([marked_paragraph], do_sample=False, num_beams=4)

    for d in gen_out[0]:
        print("---")
        print(d["output_text"])
    exit()

    # gen = GeneratorHF(model_card="facebook/bart-base", starter_file="qgen/bartb_squad_aaware_logprob_1.531.bin")
    # paragraph = "asteroid soil samples \n A small capsule containing asteroid soil samples that was dropped from 136,700 miles in space by Japan's Hayabusa2 spacecraft landed as planned in the Australian Outback on December 6. The extremely high precision required to carry out the mission thrilled many in Japan, who said they took pride in its success."
    # questions = ["What was contained in the capsule that was dropped from 136,700 miles in space?"]

    # gen = GeneratorHF(model_card="Salesforce/mixqg-large")
    # paragraph = "asteroid soil samples \n A small capsule containing asteroid soil samples that was dropped from 136,700 miles in space by Japan's Hayabusa2 spacecraft landed as planned in the Australian Outback on December 6. The extremely high precision required to carry out the mission thrilled many in Japan, who said they took pride in its success."
    # questions = ["What was dropped from space by Japan's Hayabusa2 spacecraft?"]

    # gen = GeneratorHF(model_card="microsoft/prophetnet-large-uncased-squad-qg")
    # paragraph = "asteroid soil samples [SEP] A small capsule containing asteroid soil samples that was dropped from 136,700 miles in space by Japan's Hayabusa2 spacecraft landed as planned in the Australian Outback on December 6. The extremely high precision required to carry out the mission thrilled many in Japan, who said they took pride in its success."
    # questions = ["what was in the capsule that landed in australia?"]

    # gen = GeneratorHF(model_card="gpt2-medium", starter_file="qgen/gpt2m_nf_squad_aaware_1.423.bin")
    # paragraph = "asteroid soil samples \n A small capsule containing asteroid soil samples that was dropped from 136,700 miles in space by Japan's Hayabusa2 spacecraft landed as planned in the Australian Outback on December 6. The extremely high precision required to carry out the mission thrilled many in Japan, who said they took pride in its success."
    # questions = ["What was contained in the capsule that was dropped from 136,700 miles in space?"]

    # paragraphs = [paragraph] * len(questions)

    # gen.model.eval()
    # print(gen.score(paragraphs, questions))

    # for d in gen.generate([paragraph], num_beams=1, max_gen_length=40, compute_logprobs=True):
    #     print(d[0]["output_text"])

    # print("---------")
    # tokenized_paragraphs = [torch.LongTensor(gen.tokenizer.encode(text=p)) for p in paragraphs]
    # encoder_ids = torch.nn.utils.rnn.pad_sequence(tokenized_paragraphs, batch_first=True, padding_value=0).to(gen.device)

    # tokenized_questions = [gen.tokenizer.encode(text=q, add_special_tokens=False) for q in questions]
    # decoder_input_ids = torch.nn.utils.rnn.pad_sequence([torch.LongTensor([gen.start_id] + q) for q in tokenized_questions], batch_first=True, padding_value=gen.end_id).to(gen.device)
    # decoder_output_ids = torch.nn.utils.rnn.pad_sequence([torch.LongTensor(q + [gen.end_id]) for q in tokenized_questions], batch_first=True, padding_value=-1).to(gen.device)

    # print("=============")
    # print("Likelihood function")
    # print(decoder_input_ids.tolist())
    # print(decoder_output_ids.tolist())

    # print("============")

    # model_output = gen.model(input_ids=encoder_ids, decoder_input_ids=decoder_input_ids, return_dict=True)
    # selected_logprobs = utils_rl.select_logprobs(model_output.logits, decoder_output_ids.tolist(), eos_id=gen.end_id)
    # print("Manual selected logprobs", selected_logprobs)

    # crit = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction="none")
    # N, seqlength, vocab_size = model_output.logits.shape
    # loss_components = crit(model_output.logits.view(N*seqlength, vocab_size), decoder_output_ids.contiguous().view(-1)).reshape(N, seqlength)

    # num_words = torch.sum(decoder_output_ids != -1, dim=1)
    # score_per_item = (- torch.sum(loss_components, dim=1) / num_words).tolist()

    # print("Manual score per item:", score_per_item)

    gen = GeneratorHF(model_card="distilgpt2", starter_file="qgen/dgpt2_squad_aaware_1.794.bin")

    answers = ["A small capsule", "asteroid soil samples", "136,700 miles", "Australian Outback"]
    original = "A small capsule containing asteroid soil samples that was dropped from 136,700 miles in space by Japan's Hayabusa2 spacecraft landed as planned in the Australian Outback on December 6. The extremely high precision required to carry out the mission thrilled many in Japan, who said they took pride in its success."
    paragraphs = ["%s \n %s" % (answer, original) for answer in answers]

    gen_params = [{"num_beams": 3, "num_runs": 1}, {"num_beams": 1, "num_runs": 3, "do_sample": True}]
    for gen_param in gen_params:
        print("===============")
        print(gen_param)
        batch_outs1 = gen.generate(paragraphs, max_length=100, compute_logprobs=True, **gen_param)
        for ans, outs1 in zip(answers, batch_outs1):
            print("=========")
            print("Target answer:", ans)
            for out1 in outs1:
                print("[%.3f] %s" % (out1["logprob"], out1["output_text"]))
            print("--------")

    print("========================")
    print("========================")
    print("========================")
