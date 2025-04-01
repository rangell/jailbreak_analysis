import argparse
import glob
import pathlib

from datasets import load_dataset
from strong_reject.jailbreaks import decode_dataset
from strong_reject.evaluate import evaluate_dataset


DATA_DIR = pathlib.Path(__file__).parent.resolve() / "data"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shard-dir",
        default="",
        help="Directory where JSON shards are stored"
    )
    parser.add_argument(
        "--target-model",
        default="llama3-8b",     # see config.py for a whole list of target models
        help="Name of target model."
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    shard_filenames = glob.glob(f"{args.shard_dir}/{args.target_model}*.json")

    # rangell: HACK
    if "benign" in args.shard_dir:
        args.target_model = f"benign-{args.target_model}"

    with open(f"{DATA_DIR}/all_responses-{args.target_model}.json", "w") as outfile:
        for fname in shard_filenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)

    dataset = load_dataset("json", data_files=f"{DATA_DIR}/all_responses-{args.target_model}.json")["train"]

    dataset = decode_dataset(dataset)

    dataset = evaluate_dataset(dataset, ["strongreject_rubric"])

    dataset.to_json(f"{DATA_DIR}/eval_all_responses-{args.target_model}.json")