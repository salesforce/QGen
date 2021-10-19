# MixQG: Neural Question Generation with Mixed Answer Types

This is the official code base for the following paper from Salesforce Research:

**Title**: [MixQG: Neural Question Generation with Mixed Answer Types](https://arxiv.org/abs/2110.08175)

**Authors**: Lidiya Murakhovs'ka, Chien-Sheng Wu, Tong Niu, Wenhao Liu, Caiming Xiong

## Abstract

Asking good questions is an essential ability for both human and machine intelligence. However, existing neural question generation approaches mainly focus on the short factoid type of answers. In this paper, we propose a neural question generator, MixQG, to bridge this gap. We combine 9 question answering datasets with diverse answer types, including yes/no, multiple-choice, extractive, and abstractive answers, to train a single generative model. We show with empirical results that our model outperforms existing work in both seen and unseen domains and can generate questions with different cognitive levels when conditioned on different answer types. Our code is released and well-integrated with the Huggingface library to facilitate various downstream applications.

## Usage

MixQG pre-trained models are available through the Huggingface library:

```
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "Salesforce/mixqg-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def run_qg(input_text, **generator_args):
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    generated_ids = model.generate(input_ids, **generator_args)
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
```

Input text should be formatted as follows: `f"{answer} \\n {context}"`

For example,
```
run_qg("Robert Boyle \\n In the late 17th century, Robert Boyle proved that air is necessary for combustion.")
# should output ['Who proved that air is necessary for combustion?']
```

## Released Model Checkpoints

We have released the following checkpoints for pre-trained models described in our paper:
- MixQG-base (220M parameters): [link](https://huggingface.co/Salesforce/mixqg-base)
- MixQG-large (770M parameters): [link](https://huggingface.co/Salesforce/mixqg-large)
- MixQG-3B (3B parameters): [link](https://huggingface.co/Salesforce/mixqg-3b)

## Set up
`pip install -r requirements.txt`

## Preprocessing
Preprocess the required datasets and merge them into one in the `DIR` folder.
```
DIR=/PATH/TO/DATASET/FOLDER
python data/preprocess_datasets.py --dir $DIR
python data/merge_datasets.py --dir $DIR
```
The `DIR` folder will contain each of the preprocessed in-domain and out-of-domain datasets as well as the final `mixqg` dataset.

## Training
```
num_gpus=4
model_name=t5-base
dataset=${DIR}/mixqg
output_dir=mixqg-base
lr=3e-5
bs=32

./train.sh $num_gpus $model_name $dataset $output_dir $lr $bs
```
## Fine-tuning
```
num_gpus=4
model_name=Salesforce/mixqg-base
dataset=${DIR}/squad
output_dir=mixqg-base-squad
lr=3e-6
bs=32

./train.sh $num_gpus $model_name $dataset $output_dir $lr $bs
```

## Evaluation
```
gpu=0
model=Salesforce/mixqg-base
dataset=${DIR}/squad
output_dir=mixqg-base-squad-eval
bs=32

./eval.sh $gpu $model $dataset $output_dir $bs
```

## Citation

```
@misc{murakhovska2021mixqg,
      title={MixQG: Neural Question Generation with Mixed Answer Types}, 
      author={Lidiya Murakhovs'ka and Chien-Sheng Wu and Tong Niu and Wenhao Liu and Caiming Xiong},
      year={2021},
      eprint={2110.08175},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```