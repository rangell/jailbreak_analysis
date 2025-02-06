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
from sklearn.neighbors import kneighbors_graph as knn_graph
import seaborn as sns
from tqdm import tqdm

from repe import repe_pipeline_registry # register 'rep-reading' and 'rep-control' tasks into Hugging Face pipelines

from utils import load_model_and_tokenizer


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

    parser.add_argument("--target-model", default="llama3-8b", help="Name of target model.")
    parser.add_argument("--dataset-name", default="tatsu-lab/alpaca")
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", default="/scratch/rca9780/llm-adaptive-attacks-data/knn_graphs/")

    args = parser.parse_args()

    print("args: ", vars(args))

    # register RepE pipelines
    repe_pipeline_registry()

    # for determinism, maybe need more?
    random.seed(42)

    # load the target model and its tokenizer
    model, tokenizer = load_model_and_tokenizer(args.target_model)

    # load some data
    assert args.dataset_name == "tatsu-lab/alpaca", "We only support alpaca for now"
    print(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name, split=f"train[:{args.num_samples}]")

    # format the prompts from alpaca
    full_prompts = []
    for example in dataset:
        if example.get("input", "") != "":
            chat = [{"role": "user", "content": example["instruction"] + ": " + example["input"] + "."}]
        else:
            chat = [{"role": "user", "content": example["instruction"]}]
        full_prompts.append(tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True))

    # declare the pipeline
    rep_reading_pipeline =  pipeline("rep-reading", model=model, tokenizer=tokenizer, device_map="auto")

    # get the reps
    hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))
    rep_token = -1
    reps = rep_reading_pipeline(full_prompts, rep_token=rep_token, hidden_layers=hidden_layers, batch_size=args.batch_size)
    reps = reformat_reps(reps)

    G = {layer_idx : knn_graph(layer_reps, args.k, mode='connectivity', include_self=False) for layer_idx, layer_reps in reps.items()}
    
    knn_graph_data = {
        "knn_graphs": G,
        "k": args.k,
        "prompts": full_prompts,
        "model_name": args.target_model,
        "reps": reps,
        "rep_token": rep_token,
        "hidden_layers": hidden_layers,
    }   

    with open(f'{args.output_dir}/{args.target_model}-{args.dataset_name.split("/")[1]}-knn_graph-{args.num_samples}-{args.k}.pkl', 'wb') as f:
        pickle.dump(knn_graph_data, f)