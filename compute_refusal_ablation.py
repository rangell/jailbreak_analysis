import argparse
import copy
import os
import gc
import json
import glob
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn.functional as F
from tqdm import tqdm

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import load_model_and_tokenizer, get_all_hidden_states, batch_generator


def compute_refusal_ablation(args, model, tokenizer, refusal_directions, evals_dataset_elt):
    # Compute original prompt hidden states
    original_prompts_list = [tokenizer.apply_chat_template([{"role": "user", "content": evals_dataset_elt["forbidden_prompt"]}], tokenize=False, add_generation_prompt=True)]
    original_hidden_states = []
    for batch_prompts in batch_generator(original_prompts_list, batch_size=1):
        original_hidden_states.append(get_all_hidden_states(model, tokenizer, batch_prompts)[:, -5:, :, :])
    original_hidden_states = torch.cat(original_hidden_states, dim=0).float()

    # Compute jailbreak prompt hidden states
    jailbreak_prompt_convo = evals_dataset_elt["jailbroken_prompt"]
    jailbreak_prompts_list = [tokenizer.apply_chat_template(jailbreak_prompt_convo, tokenize=False, add_generation_prompt=True)]
    jailbreak_hidden_states = []
    for batch_prompts in batch_generator(jailbreak_prompts_list, batch_size=1):
        jailbreak_hidden_states.append(get_all_hidden_states(model, tokenizer, batch_prompts)[:, -5:, :, :])
    jailbreak_hidden_states = torch.cat(jailbreak_hidden_states, dim=0).float()

    harmfulness_scores = torch.einsum("btld,tld->btl", jailbreak_hidden_states, refusal_directions)[0].cpu().numpy().tolist()
    harmfulness_ablation_scores = torch.einsum("btld,tld->btl", original_hidden_states - jailbreak_hidden_states, refusal_directions)[0].cpu().numpy().tolist()

    evals_dataset_elt["harmfulness_scores"] = harmfulness_scores
    evals_dataset_elt["harmfulness_ablation_scores"] = harmfulness_ablation_scores

    gc.collect()
    torch.cuda.empty_cache()

    return evals_dataset_elt


def main(args):
    # Setup model
    model, tokenizer = load_model_and_tokenizer(args.target_model)

    # load the dataset
    evals_dataset = load_dataset("json", data_files=args.evaluated_responses)["train"]

    # Grab all refusal directions
    with open(f"refusal_directions-{args.target_model}.pkl", "rb") as f:
        refusal_directions = torch.from_numpy(pickle.load(f)[args.target_model]).to("cuda")

    scored_data = []
    for x in tqdm(evals_dataset):
        scored_data.append(compute_refusal_ablation(args, model, tokenizer, refusal_directions, x))

    scored_dataset = Dataset.from_pandas(pd.DataFrame(data=scored_data))
    scored_dataset.to_json(f"{os.path.dirname(args.evaluated_responses)}/refusal_plus_{os.path.basename(args.evaluated_responses)}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", type=str, default="llama3-8b")
    parser.add_argument("--evaluated-responses", type=str, help="evaluated responses JSON file")
    args = parser.parse_args()

    main(args)