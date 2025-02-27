import argparse
import os
import pathlib
import json

from datasets import load_dataset, Dataset
from dotenv import load_dotenv

from strong_reject.evaluate import evaluate_dataset


parser = argparse.ArgumentParser()
parser.add_argument("--responses_path", required=True, help="path to JSON-formatted responses.")


def main(args):
    # load the dataset
    dataset = load_dataset("json", data_files=args.responses_path)["train"]

    # renaming so we can use the strongreject api
    harmful_responses = [x["model_responses"]["ablated"] for x in dataset]
    dataset = dataset.add_column("response", harmful_responses)

    # evaluate the dataset
    dataset = evaluate_dataset(dataset, ["strongreject_rubric"])

    # dump the evaluated dataset
    dataset.to_json(f"evals_{args.responses_path}")


if __name__ == "__main__":
    load_dotenv()
    args = parser.parse_args()

    main(args)