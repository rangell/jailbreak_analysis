import argparse
from argparse import Namespace
from collections import defaultdict
import copy
import gc
import os
import glob
import json
import time
import itertools
import math
import random
import pickle
import io
import gzip
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
import joblib

from utils import load_model_and_tokenizer, judge_rule_based


def custom_collate_fn(data):
    batched_data = defaultdict(list)
    for datum in data:
        for k, v in datum.items():
            batched_data[k].append(v)
    return batched_data


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--target-model", default="llama3-8b", help="Name of target model (see `conversers.py` for all the available models).")
    parser.add_argument("--jailbreak-dataset", default="jailbreak_success", help="Jailbreak dataset to use.", choices=["jailbreak_success", "wildjailbreak"])
    parser.add_argument("--max-n-tokens", type=int, default=150, help="Number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.6, help="Temperature for generation.")
    parser.add_argument("--top-k", type=int, default=50, help="Top-K for generation.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p for generation.")
    parser.add_argument("--num-return-sequences", type=int, default=10, help="Number of sequences to sample from the model.")
    parser.add_argument("--batch-size", type=int, default=16, help="Max batch size for generation.")
    parser.add_argument("--output-dir", help="Path to output directory.", required=True)
    args = parser.parse_args()

    print("args: ", vars(args))

    # for determinism, maybe need more?
    random.seed(42)

    # load the target model and its tokenizer
    model, tokenizer = load_model_and_tokenizer(args.target_model)

    # do some size-based sorting for optimal generation
    prompt_key = "jailbreak_prompt_text"
    with open(f"data/{args.jailbreak_dataset}_preprocessed.pkl", "rb") as f:
        _jailbreak_dataset = pickle.load(f)
    sized_examples = []
    for x in _jailbreak_dataset:
        x[prompt_key] = tokenizer.apply_chat_template(x["jailbreak_prompt_convo"], tokenize=False, add_generation_prompt=True)
        num_tokens = len(tokenizer(x[prompt_key])["input_ids"])
        sized_examples.append((num_tokens, x))
    sized_examples = sorted(sized_examples, key=lambda x : x[0])
    jailbreak_dataset = [x[1] for x in sized_examples]
    jailbreak_dataloader = DataLoader(jailbreak_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)

    # recover from previous run
    completed_shards = glob.glob(f"{args.output_dir}/{args.target_model}*.pkl")
    max_shard_index = max([int(shard_fname.split("-")[-1].split(".")[0]) for shard_fname in completed_shards])

    subbatch_factor = 1
    idx = 0
    for batch in tqdm(jailbreak_dataloader):

        # wait until we get to where we left off last time
        if idx <= max_shard_index:
            idx += 1
            continue

        _batch = copy.deepcopy(batch)
        _batch["jailbreak_prompt_tokenized"] = tokenizer.apply_chat_template(
            _batch["jailbreak_prompt_convo"],
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            return_tensors="pt").cuda()
        input_len = _batch["jailbreak_prompt_tokenized"].shape[1]
        batch_processed = False
        while not batch_processed:
            try:
                responses = []
                subbatch_size = args.batch_size // subbatch_factor
                batch_length = len(batch["prompt_name"])
                for i in range(math.ceil(batch_length / subbatch_size)):
                    start_idx = i*subbatch_size
                    end_idx = min((i+1)*subbatch_size, batch_length)
                    output = model.generate(
                        _batch["jailbreak_prompt_tokenized"].cuda(),
                        max_new_tokens=args.max_n_tokens,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,  # added for Mistral
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        num_return_sequences=args.num_return_sequences)
                    _subbatch_responses = tokenizer.batch_decode(output[:, input_len:], skip_special_tokens=True)
                    subbatch_responses = [_subbatch_responses[j*args.num_return_sequences:(j+1)*args.num_return_sequences] for j in range(end_idx - start_idx)]
                    responses += subbatch_responses
                _batch["responses"] = responses
                batch_processed = True
            except torch.OutOfMemoryError as e: 
                gc.collect()
                torch.cuda.empty_cache()
                if subbatch_factor == args.batch_size:
                    raise ValueError("Subbatch size is already 1!!!")
                subbatch_factor *= 2
                print("new subbatch factor: ", subbatch_factor)
        _batch['jailbroken_rule_judge'] = [[judge_rule_based(resp) for resp in responses] for responses in _batch['responses']]

        prompt_responses = {}
        for i in range(len(_batch["prompt_name"])):
            example_dict = {}
            for key in _batch.keys():
                example_dict[key] = _batch[key][i]
            prompt_responses[(_batch["jailbreak_prompt_name"][i], _batch["original_prompt_text"][i])] = example_dict

        del _batch

        latest_pickle_fname = f'{args.output_dir}/{args.target_model}-responses-{idx}.pkl'
        with open(latest_pickle_fname, 'wb') as f:
            pickle.dump(prompt_responses, f, protocol=5)

        del prompt_responses

        idx += 1