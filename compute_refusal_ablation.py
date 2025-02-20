import argparse
import copy
import os
import gc
import json
import glob
import numpy as np
import pickle
import torch
import torch.nn.functional as F
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import load_model_and_tokenizer, get_all_hidden_states


def augment_shard_with_refusal_ablation(args, model, tokenizer, refusal_directions, shard_fname):
    # Augment shard data with refusal information
    with open(shard_fname, "rb") as f:
        shard_data = pickle.load(f)
    
    # Convert the data from at dictionary to a list
    shard_dataset = list(shard_data.items())

    # Compute original prompt hidden states
    original_prompts_list = [tokenizer.apply_chat_template([{"role": "user", "content": x[1]["original_prompt_text"]}], tokenize=False, add_generation_prompt=True) for x in shard_dataset]
    original_hidden_states = get_all_hidden_states(model, tokenizer, original_prompts_list)

    # Compute jailbreak prompt hidden states
    jailbreak_prompts_list = [tokenizer.apply_chat_template(x[1]["jailbreak_prompt_convo"], tokenize=False, add_generation_prompt=True) for x in shard_dataset]
    jailbreak_hidden_states = get_all_hidden_states(model, tokenizer, jailbreak_prompts_list)

    from IPython import embed; embed(); exit()

    # TODO: compute harmful direction scores

    for idx, (k, v) in enumerate(shard_dataset):
        _v = copy.deepcopy(v)
        _v["refusal_score2"] = jailbreak_scores[idx].item()
        shard_data[k] = _v

    with open(shard_fname, "wb") as f:
        pickle.dump(shard_data, f)

    gc.collect()
    torch.cuda.empty_cache()


def main(args):
    # Setup model
    model, tokenizer = load_model_and_tokenizer(args.target_model)

    # Grab all refusal directions
    with open("refusal_directions.pkl", "rb") as f:
        refusal_directions = pickle.load(f)[args.target_model]

    # Augment shards
    for shard_fname in tqdm(glob.glob(f"{args.responses_dir}/{args.target_model}*.pkl")):
        augment_shard_with_refusal_ablation(args, model, tokenizer, refusal_directions, shard_fname)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", type=str, default="llama3-8b")
    parser.add_argument("--responses-dir", type=str, default="/scratch/rca9780/jailbreak_analysis_data/jailbreak_responses/jailbreak_success/")
    args = parser.parse_args()

    main(args)