import argparse
import os
import pathlib

from datasets import concatenate_datasets
from dotenv import load_dotenv

from strong_reject.load_datasets import load_strongreject
from strong_reject.evaluate import evaluate_dataset
from strong_reject.generate import generate_to_dataset
from strong_reject.jailbreaks import (
    apply_jailbreaks_to_dataset,
    decode_dataset,
    registered_jailbreaks,
)

from utils import get_text_generation_pipeline

DATA_DIR = pathlib.Path(__file__).parent.resolve() / "data"

parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true")
parser.add_argument(
    "--target-model",
    default="llama3-8b",     # see config.py for a whole list of target models
    help="Name of target model."
)


def main():
    if os.getenv("TESTING"):
        dataset = load_strongreject().shuffle().select(range(2))
        jailbreaks = "base64", "pair"
        models = (
            get_text_generation_pipeline(args.target_model),
        )
        max_tokens = 100
    else:
        dataset = load_strongreject()
        jailbreaks = registered_jailbreaks
        jailbreaks = {k: v for k, v in registered_jailbreaks.items() if "translation" not in k}
        models = (get_text_generation_pipeline(args.target_model),)
        max_tokens = 384

    dataset = apply_jailbreaks_to_dataset(dataset, jailbreaks)
    dataset.to_json(str(DATA_DIR / "interim" / "jailbreaks.json"))
    datasets = []
    for model in models:
        model_name = model if isinstance(model, str) else model.model.name_or_path
        model_dataset = generate_to_dataset(
            dataset,
            models=[model],
            target_column="jailbroken_prompt",
            max_new_tokens=max_tokens,
            do_sample=True,
            num_return_sequences=10,
            temperature=0.6,
            top_k=50,
            top_p=0.95
        )
        model_dataset.to_json(str(DATA_DIR / "interim" / f"responses_{args.target_model}.json"))
        model_dataset = evaluate_dataset(model_dataset, ["strongreject_rubric"])
        model_dataset.to_json(str(DATA_DIR / "interim" / f"evals_{args.target_model}.json"))
        datasets.append(model_dataset)

    return concatenate_datasets(datasets)


if __name__ == "__main__":
    load_dotenv()
    filename = "evals.json"
    args = parser.parse_args()
    if args.test:
        os.environ["TESTING"] = "true"
        filename = "evals_test.json"

    dataset = main()
    dataset.to_json(str(DATA_DIR / "interim" / filename))
