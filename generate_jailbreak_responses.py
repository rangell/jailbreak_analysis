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
import torch.nn.functional as F
from torch.utils.data import DataLoader
import joblib

from utils import load_model_and_tokenizer, judge_rule_based


def custom_collate_fn(data):
    batched_data = defaultdict(list)
    for datum in data:
        for k, v in datum.items():
            batched_data[k].append(v)
    return batched_data


def concatenate_with_padding(tensors, dim=0, pad_value=0):
    """Concatenates a list of tensors with padding.

    Args:
        tensors (list of torch.Tensor): List of tensors to concatenate.
        dim (int): Dimension to concatenate along.
        pad_value (int): Value to use for padding.

    Returns:
        torch.Tensor: Concatenated tensor.
    """
    max_size = max([t.size(d) if d != dim else 0 for t in tensors for d in range(t.ndim)])
    padded_tensors = []
    for tensor in tensors:
        for d in range(tensor.ndim):
            if d == dim:
                continue
            pad_size = max_size - tensor.size(d)
            if pad_size > 0:
                pad_before = [0] * (tensor.ndim * 2)
                pad_before[-(2 * (tensor.ndim - dim) - 1)] = pad_size
                padded_tensor = F.pad(tensor, pad_before, mode='constant', value=pad_value)
            else:
                padded_tensor = tensor
        padded_tensors.append(padded_tensor)

    return torch.cat(padded_tensors, dim=dim)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--target-model", default="llama3-8b", help="Name of target model (see `conversers.py` for all the available models).")
    parser.add_argument("--jailbreak-dataset", default="jailbreak_success", help="JSON-formatted jailbreak dataset to use.")
    parser.add_argument("--max-new-tokens", type=int, default=150, help="Number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.6, help="Temperature for generation.")
    parser.add_argument("--top-k", type=int, default=50, help="Top-K for generation.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p for generation.")
    parser.add_argument("--num-return-sequences", type=int, default=10, help="Number of sequences to sample from the model.")
    parser.add_argument("--batch-size", type=int, default=64, help="Max batch size for generation.")
    parser.add_argument("--output-dir", help="Path to output directory.", required=True)
    args = parser.parse_args()

    print("args: ", vars(args))

    # for determinism, maybe need more?
    random.seed(42)

    # load the target model and its tokenizer
    model, tokenizer = load_model_and_tokenizer(args.target_model)

    # load the jailbreak dataset
    _jailbreaks_dataset = load_dataset("json", data_files=args.jailbreak_dataset)["train"]

    # do some size-based sorting for optimal generation
    prompt_key = "jailbreak_prompt_text"
    sized_examples = []
    for x in _jailbreaks_dataset:
        x[prompt_key] = tokenizer.apply_chat_template(x["jailbroken_prompt"], tokenize=False, add_generation_prompt=True)
        num_tokens = len(tokenizer(x[prompt_key])["input_ids"])
        sized_examples.append((num_tokens, x))
    sized_examples = sorted(sized_examples, key=lambda x : x[0])
    jailbreak_dataset = [x[1] for x in sized_examples]
    jailbreak_dataloader = DataLoader(jailbreak_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)

    # recover from previous run
    completed_shards = glob.glob(f"{args.output_dir}/{args.target_model}*.json")
    if len(completed_shards) > 0:
        max_shard_index = max([int(shard_fname.split("-")[-1].split(".")[0]) for shard_fname in completed_shards])
    else:
        max_shard_index = -1

    subbatch_factor = 1
    idx = 0
    for batch in tqdm(jailbreak_dataloader):

        # wait until we get to where we left off last time
        if idx <= max_shard_index:
            idx += 1
            continue

        _batch = copy.deepcopy(batch)
        _text = tokenizer.apply_chat_template(_batch["jailbroken_prompt"], tokenize=False, add_generation_prompt=True)

        batch_processed = False
        while not batch_processed:
            try:
                responses = []
                subbatch_size = args.batch_size // subbatch_factor
                batch_length = len(batch["jailbreak"])
                for i in range(math.ceil(batch_length / subbatch_size)):
                    start_idx = i*subbatch_size
                    end_idx = min((i+1)*subbatch_size, batch_length)
                    inputs = tokenizer(_text[start_idx:end_idx], padding=True, return_tensors="pt").to("cuda")
                    input_len = inputs["input_ids"].shape[1]
                    output_sequences = None
                    if subbatch_size == 1:
                        # For very long sequences, we might need to break up `num_return_sequences`
                        _output_sequences = []
                        for _ in range(args.num_return_sequences):
                            output = model.generate(
                                **inputs,
                                max_new_tokens=args.max_new_tokens,
                                eos_token_id=tokenizer.eos_token_id,
                                pad_token_id=tokenizer.pad_token_id,  # added for Mistral
                                do_sample=True,
                                temperature=args.temperature,
                                top_p=args.top_p,
                                top_k=args.top_k,
                                return_dict_in_generate=True,
                                num_return_sequences=1)
                            _output_sequences.append(output.sequences)
                        output_sequences = concatenate_with_padding(_output_sequences, dim=0, pad_value=tokenizer.pad_token_id)
                    else:
                        output = model.generate(
                            **inputs,
                            max_new_tokens=args.max_new_tokens,
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.pad_token_id,  # added for Mistral
                            do_sample=True,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            top_k=args.top_k,
                            return_dict_in_generate=True,
                            num_return_sequences=args.num_return_sequences)
                        output_sequences = output.sequences
                    _subbatch_responses = tokenizer.batch_decode(output_sequences[:, input_len:], skip_special_tokens=True)
                    subbatch_responses = [_subbatch_responses[j*args.num_return_sequences:(j+1)*args.num_return_sequences] for j in range(end_idx - start_idx)]
                    responses += subbatch_responses
                _batch["response"] = responses
                batch_processed = True
            except torch.OutOfMemoryError as e: 
                gc.collect()
                torch.cuda.empty_cache()
                if subbatch_factor == args.batch_size:
                    raise ValueError("Subbatch size is already 1!!!")
                subbatch_factor *= 2
                print("new subbatch factor: ", subbatch_factor)
        _batch['jailbroken_rule_judge'] = [[judge_rule_based(resp) for resp in responses] for responses in _batch['response']]

        latest_shard_fname = f'{args.output_dir}/{args.target_model}-responses-{idx}.json'
        with open(latest_shard_fname, 'w') as f:
            for i in range(args.batch_size):
                f.write(json.dumps({k : v[i] for k, v in _batch.items()}) + "\n")

        idx += 1