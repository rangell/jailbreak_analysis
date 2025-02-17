import argparse
import os
import json
import glob
import numpy as np
import pickle
import torch
import torch.nn.functional as F
from tqdm import tqdm

from nnsight import NNsight
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import expand_shortcut_model_name


def get_activations(args, model, tokenizer, dataset):
    jailbreak_activations = []
    original_activations = []
    for example in tqdm(dataset):
        # Compute jailbreak residual stream activations at specified location
        instruction = tokenizer.apply_chat_template(
            example[1]["jailbreak_prompt_convo"],
            add_generation_prompt=True,
            tokenize=False
        )
        #inputs = tokenizer(instruction, return_tensors="pt").input_ids  
        #acts = []
        #with model.trace(inputs) as trace:
        #    for l in range(model.config.num_hidden_layers):
        #        acts.append(model.model.layers[l].output.save())
        #acts = torch.cat([a[0] for a in acts], dim=0)
        inputs = tokenizer(instruction, return_tensors="pt").input_ids  
        with model.trace(inputs) as trace:
            acts = model.model.layers[args.layer].output.save()
        if isinstance(acts, tuple):
            acts = acts[0]
        jailbreak_activations.append(acts[:, args.position].detach().cpu())

        # Compare with original instruction
        instruction = tokenizer.apply_chat_template(
            [{"role": "user", "content": example[1]["original_prompt_text"]}],
            add_generation_prompt=True,
            tokenize=False
        )
        inputs = tokenizer(instruction, return_tensors="pt").input_ids  
        #acts = []
        #with model.trace(inputs) as trace:
        #    for l in range(model.config.num_hidden_layers):
        #        acts.append(model.model.layers[l].output.save())
        #acts = torch.cat([a[0] for a in acts], dim=0)
        #original_activations.append(acts[:, args.position, :].detach().cpu())
        with model.trace(inputs) as trace:
            acts = model.model.layers[args.layer].output.save()
        if isinstance(acts, tuple):
            acts = acts[0]
        original_activations.append(acts[:, args.position].detach().cpu())
    #return torch.stack(jailbreak_activations), torch.stack(original_activations)
    return torch.cat(jailbreak_activations, dim=0), torch.cat(original_activations, dim=0)


def augment_shard_with_refusal_ablation(args, model, tokenizer, refusal_direction_dict, shard_fname):
    # Normalize the refusal direction
    refusal_direction = F.normalize(refusal_direction_dict["best_refusal_direction"], dim=0)

    # Set the best refusal direction position and layer for this model
    args.position = refusal_direction_dict["best_position"]
    args.layer = refusal_direction_dict["best_layer"]

    # Augment shard data with refusal information
    with open(shard_fname, "rb") as f:
        shard_data = pickle.load(f)
    
    # Convert the data from at dictionary to a list
    shard_dataset = list(shard_data.items())

    # Get the activations at the best refusal direction location
    jailbreak_activations, original_activations = get_activations(args, model, tokenizer, shard_dataset)

    from IPython import embed; embed(); exit()

    # TODO: compute logits for next token prediction
    # TODO: record vector at position, project onto output space maybe quantize?
    shard_data = {k: v for k, v in shard_dataset}

    with open(shard_fname, "wb") as f:
        pickle.dump(shard_data, f)


def main(args):
    # Setup model
    access_token = os.environ['HF_TOKEN']
    tokenizer = AutoTokenizer.from_pretrained(
        expand_shortcut_model_name(args.model_name), 
        token=access_token, 
        padding_side="left"
    )
    model = AutoModelForCausalLM.from_pretrained(
        expand_shortcut_model_name(args.model_name), 
        token=access_token,
        device_map="auto"
    )
    model = NNsight(model)

    # Grab the refusal direction
    with open(f"{args.refusal_directions_dir}/{args.model_name}.pkl", "rb") as f:
        refusal_direction_dict = pickle.load(f)

    # Augment shards
    for shard_fname in tqdm(glob.glob(f"{args.responses_dir}/{args.model_name}*.pkl")):
        augment_shard_with_refusal_ablation(args, model, tokenizer, refusal_direction_dict, shard_fname)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--responses_dir", type=str, default="/scratch/rca9780/llm-adaptive-attacks-data/jailbreak_responses/jailbreak_success/")
    parser.add_argument("--refusal_directions_dir", type=str, default="outputs/")
    args = parser.parse_args()

    main(args)
