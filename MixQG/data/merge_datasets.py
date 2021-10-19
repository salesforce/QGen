import argparse
import os

from datasets import DatasetDict, load_from_disk, concatenate_datasets


def main(args):
    DIR = args.dir

    # Load datasets
    mrqa = load_from_disk(f"{DIR}/mrqa")
    narrativeqa = load_from_disk(f"{DIR}/narrativeqa")
    mctest = load_from_disk(f"{DIR}/mctest")
    boolq = load_from_disk(f"{DIR}/boolq")

    loaded_datasets = [mrqa, narrativeqa, mctest, boolq]

    # Shuffle
    train_datasets = [d["train"].shuffle()
                    for d in loaded_datasets if "train" in d.keys()]
    eval_datasets = [d["validation"]
                    for d in loaded_datasets if "validation" in d.keys()]
    test_datasets = [d["test"]
                    for d in loaded_datasets if "test" in d.keys()]

    # Merge & Save
    train_dataset = concatenate_datasets(train_datasets)
    eval_dataset = concatenate_datasets(eval_datasets)
    test_dataset = concatenate_datasets(test_datasets)

    combined = DatasetDict({
        "train": train_dataset.shuffle(),
        "validation": eval_dataset,
        "test": test_dataset
    })

    if not os.path.isdir(f"{DIR}/mixqg"):
        combined.save_to_disk(f"{DIR}/mixqg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="",
                        help="Path to the datasets directory.")
    args = parser.parse_args()
    main(args)
