import argparse
from argparse import Namespace
from collections import defaultdict
import copy
import os
import json
import itertools
import random
import pickle
import numpy as np
from scipy.stats import multivariate_normal
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
import seaborn as sns
from tqdm import tqdm

#from repe import repe_pipeline_registry # register 'rep-reading' and 'rep-control' tasks into Hugging Face pipelines

from utils import load_model_and_tokenizer, batch_generator, get_all_hidden_states


def reformat_reps(orig_reps):
    _per_layer_dict = {}
    for k in orig_reps[0].keys():
        _per_layer_dict[k] = torch.concat([x[k] for x in orig_reps])
    out_reps = _per_layer_dict
    for k, reps in out_reps.items():
        out_reps[k] = reps.numpy()
    return out_reps


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target-model",
        default="llama3-8b",     # see conversers.py for a whole list of target models
        help="Name of target model."
    )
    args = parser.parse_args()

    print("args: ", vars(args))

    # for determinism, maybe need more?
    random.seed(42)

    ## register RepE pipelines
    #repe_pipeline_registry()

    # load the target model and its tokenizer
    model, tokenizer = load_model_and_tokenizer(args.target_model)
    
    # load harmless and harmful datasets
    alpaca_dataset = load_dataset("tatsu-lab/alpaca", split="train")
    advbench_dataset = load_dataset("walledai/AdvBench", split="train")

    # convert data into full prompts
    harmless_prompts = []
    for example in alpaca_dataset:
        if example.get("input", "") != "":
            chat = [{"role": "user", "content": example["instruction"] + ": " + example["input"] + "."}]
        else:
            chat = [{"role": "user", "content": example["instruction"]}]
        harmless_prompts.append(tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True))
    
    harmful_prompts = []
    for example in advbench_dataset:
        chat = [{"role": "user", "content": example["prompt"]}]
        harmful_prompts.append(tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True))

    # only need the first 500 examples from each
    harmless_prompts = harmless_prompts[:500]
    harmful_prompts = harmful_prompts[:500]

    del advbench_dataset
    del alpaca_dataset

    # TODO: add this as a cmd line arg
    batch_size = 16

    # compute harmless representations
    harmless_hidden_states = []
    for prompts_batch in batch_generator(harmless_prompts, batch_size=batch_size):
        batch_hidden_states = get_all_hidden_states(model, tokenizer, prompts_batch)[:, -5:, :, :]
        harmless_hidden_states.append(batch_hidden_states)
    harmless_hidden_states = np.concatenate(harmless_hidden_states, axis=0)

    # compute harmful representations
    harmful_hidden_states = []
    for prompts_batch in batch_generator(harmful_prompts, batch_size=batch_size):
        batch_hidden_states = get_all_hidden_states(model, tokenizer, prompts_batch)[:, -5:, :, :]
        harmful_hidden_states.append(batch_hidden_states)
    harmful_hidden_states = np.concatenate(harmful_hidden_states, axis=0)

    # compute harmless direction using difference of means
    mean_harmless = np.mean(harmless_hidden_states, axis=0)
    mean_harmful = np.mean(harmful_hidden_states, axis=0)
    refusal_directions = mean_harmful - mean_harmless
    refusal_directions /= np.linalg.norm(refusal_directions, axis=-1)[:, :, None]

    refusal_directions_fname = "refusal_directions.pkl"

    if os.path.exists(refusal_directions_fname):
        with open(refusal_directions_fname, "rb") as f:
            meta_refusal_directions = pickle.load(f)
    else:
        meta_refusal_directions = {}

    meta_refusal_directions[args.target_model] = refusal_directions

    with open(refusal_directions_fname, "wb") as f:
        pickle.dump(meta_refusal_directions, f)