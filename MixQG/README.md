# MixQG

This is the official code base for the following paper from Salesforce Research:

[MixQG: Neural Question Generation with Mixed Answer Types](https://arxiv.org/abs/2110.08175)

MixQG is a new question generation model pre-trained on a collection of QA datasets with a mix of answer types.

# Usage

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

# Released Model Checkpoints

We have released the following checkpoints for pre-trained models described in our paper:
- MixQG-base (220M parameters): [link](https://huggingface.co/Salesforce/mixqg-base)
- MixQG-large (770M parameters): [link](https://huggingface.co/Salesforce/mixqg-large)
- MixQG-3B (3B parameters): [link](https://huggingface.co/Salesforce/mixqg-3b)

# Set up
`pip install -r requirements.txt`

# Preprocessing
Preprocess the required datasets and merge them into one in the `DIR` folder.
```
DIR=/PATH/TO/DATASET/FOLDER
python data/preprocess_datasets.py --dir $DIR
python data/merge_datasets.py --dir $DIR
```

# Training
```
num_gpus=4
model_name=t5-base
dataset=mixqg
output_dir=mixqg-base
lr=3e-5
bs=32

./train.sh $num_gpus $model_name $dataset $output_dir $lr $bs
```
# Fine-tuning
```
num_gpus=4
model_name=mixqg-base
dataset=squad
output_dir=mixqg-base-squad
lr=3e-6
bs=32

./train.sh $num_gpus $model_name $dataset $output_dir $lr $bs
```

# Evaluation
```
gpu=0
model=Salesforce/mixqg-base
dataset=squad
output_dir=mixqg-base-squad
bs=32

./eval.sh $gpu $model $dataset $output_dir $bs
```

# Citation

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