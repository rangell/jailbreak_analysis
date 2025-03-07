import argparse
import os
import pathlib

from datasets import load_dataset
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
    model = get_text_generation_pipeline(args.target_model)
    if args.test:
        dataset = load_strongreject().shuffle().select(range(2))
        jailbreaks = "style_injection_json", "few_shot_json"
        max_tokens = 100
    else:
        dataset = load_strongreject()
        jailbreaks = registered_jailbreaks
        jailbreaks = {k: v for k, v in registered_jailbreaks.items() if "translation" not in k}
        max_tokens = 384

    # compute the jailbreaks
    jailbreak_cache_fname = str(DATA_DIR / "interim" / f"jailbreaks_{args.target_model}.json")
    if not args.test and os.path.exists(jailbreak_cache_fname):
        jailbreak_dataset = load_dataset("json", data_files=jailbreak_cache_fname)["train"]
    else:
        jailbreak_dataset = apply_jailbreaks_to_dataset(dataset, jailbreaks)
        if not args.test:
            jailbreak_dataset.to_json(jailbreak_cache_fname)

    # compute the responses
    responses_cache_fname = str(DATA_DIR / "interim" / f"responses_{args.target_model}.json")
    if not args.test and os.path.exists(responses_cache_fname):
        responses_dataset = load_dataset("json", data_files=responses_cache_fname)["train"]
    else:
        model_name = model if isinstance(model, str) else model.model.name_or_path
        responses_dataset = generate_to_dataset(
            jailbreak_dataset,
            models=[model],
            target_column="jailbroken_prompt",
            max_new_tokens=max_tokens,
            do_sample=True,
            num_return_sequences=10,
            temperature=0.6,
            top_k=50,
            top_p=0.95
        )
        if not args.test:
            responses_dataset.to_json(responses_cache_fname)

    exit()

    # evaluate the responses (does not require GPU)
    evaluated_cache_fname = str(DATA_DIR / "interim" / f"evals_{args.target_model}.json")
    if not args.test and os.path.exists(responses_cache_fname):
        responses_dataset = load_dataset("json", data_files=responses_cache_fname)["train"]
    else:
        evaluated_dataset = evaluate_dataset(responses_dataset, ["strongreject_rubric"])
        if not args.test:
            evaluated_dataset.to_json(evaluated_cache_fname)


if __name__ == "__main__":
    load_dotenv()
    args = parser.parse_args()
    dataset = main()