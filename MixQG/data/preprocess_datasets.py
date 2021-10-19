'''
Copyright (c) 2021, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

import argparse
import os
import spacy

from datasets import load_dataset


nlp = spacy.load("en_core_web_sm")
MC_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}


def preprocess_squad(examples):
    context = []
    question = []
    answer = []
    for i in range(len(examples["answers"])):
        if len(examples["answers"][i]["text"]) > 0:
            answer.append(examples["answers"][i]["text"][0])
            context.append(examples["context"][i])
            question.append(examples["question"][i])

    return {
        "context": context,
        "question": question,
        "answer": answer
    }


def preprocess_narrative_qa(examples):
    context = []
    question = []
    answer = []
    for i in range(len(examples['answers'])):
        context.append(examples['document'][i]['summary']['text'])
        question.append(examples['question'][i]['text'])
        answer.append(examples['answers'][i][0]['text'])

    return {
        "context": context,
        "question": question,
        "answer": answer
    }


def preprocess_mrqa(examples):
    question = []
    answer = []
    context = []
    for i in range(len(examples["answers"])):
        if len(examples["answers"][i]) > 0:
            answer.append(examples["answers"][i][0])
            context.append(examples["context"][i])
            question.append(examples["question"][i])

    return {
        "context": context,
        "question": question,
        "answer": answer
    }


def preprocess_mctest(examples):
    context = examples['story']
    question = examples['question']
    answer = []
    for i in range(len(examples['question'])):
        answer_letter = examples['answer'][i]
        options = examples['answer_options'][i]
        correct_answer = options[answer_letter]
        answer.append(correct_answer)

    return {
        "context": context,
        "question": question,
        "answer": answer
    }


def preprocess_drop(examples):
    question = examples["question"]
    context = examples["passage"]
    answer = []
    for i in range(len(examples["answers_spans"])):
        answer.append(examples["answers_spans"][i]["spans"][0])

    return {
        "context": context,
        "question": question,
        "answer": answer
    }


def preprocess_boolq(examples):
    context = examples['passage']
    question = examples['question']
    answer = []
    for i in range(len(examples['question'])):
        ans = 'yes' if examples['answer'][i] else 'no'
        doc = nlp(examples['question'][i])
        entities = " ".join([ent.text for ent in doc.ents])
        if len(entities) > 0:
            answer.append(f"{ans} {entities}")
        else:
            answer.append(ans)
    return {
        "context": context,
        "question": question,
        "answer": answer
    }


def process_dataset(DIR, dataset_name, process_func):
    if os.path.isdir(f"{DIR}/{dataset_name}"):
        return
    dataset = load_dataset(dataset_name)
    column_names = dataset["train"].column_names
    processed = dataset.map(
        process_func,
        batched=True,
        num_proc=8,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc=f"Running preprocessing on {dataset_name} dataset",
    )
    print(f"Saving to disk at {DIR}/{dataset_name}")
    processed.save_to_disk(f"{DIR}/{dataset_name}")
    del processed


def mctest(DIR, dataset_name="mctest"):
    if os.path.isdir(f"{DIR}/{dataset_name}"):
        return
    dataset = load_dataset("sagnikrayc/mctest")
    column_names = dataset["train"].column_names
    processed = dataset.map(
        preprocess_mctest,
        batched=True,
        num_proc=8,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc=f"Running preprocessing on {dataset_name} dataset",
    )
    print(f"Saving to disk at {DIR}/{dataset_name}")
    processed.save_to_disk(f"{DIR}/{dataset_name}")
    del processed


def natural_questions(DIR, dataset_name="natural_questions"):
    if os.path.isdir(f"{DIR}/{dataset_name}"):
        return
    dataset = load_dataset("mrqa")
    dataset = dataset.filter(lambda ex: ex["subset"] == "NaturalQuestionsShort")
    column_names = dataset["train"].column_names
    processed = dataset.map(
        preprocess_mrqa,
        batched=True,
        num_proc=8,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc=f"Running preprocessing on {dataset_name} dataset",
    )
    print(f"Saving to disk at {DIR}/{dataset_name}")
    processed.save_to_disk(f"{DIR}/{dataset_name}")
    del processed
    

def main(args):
    DIR = args.dir

    process_dataset(DIR, "mrqa", preprocess_mrqa)
    process_dataset(DIR, "narrativeqa", preprocess_narrative_qa)
    mctest(DIR)
    process_dataset(DIR, "boolq", preprocess_boolq)

    process_dataset(DIR, "squad", preprocess_squad)
    process_dataset(DIR, "quoref", preprocess_squad)
    process_dataset(DIR, "drop", preprocess_drop)
    natural_questions(DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="",
                        help="Path to the datasets directory.")
    args = parser.parse_args()
    main(args)
