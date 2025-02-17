import argparse
import os
import json
import numpy as np
import pickle
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from nnsight import NNsight
from transformers import AutoTokenizer, AutoModelForCausalLM
from refusal_direction.pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak
from utils import expand_shortcut_model_name


def get_mean_activations(args, model, tokenizer, dataset):
    activations = []
    for example in tqdm(dataset):
        instruction = tokenizer.apply_chat_template(
            [{"role": "user", "content": example["instruction"]}],
            add_generation_prompt=True,
            tokenize=False
        )
        inputs = tokenizer(instruction, return_tensors="pt").input_ids  
        with model.trace(inputs) as trace:
            acts = model.model.layers[args.layer].output.save()
        if isinstance(acts, tuple):
            acts = acts[0]
        activations.append(acts[:, args.position].detach().cpu())
    return torch.cat(activations, dim=0)

def create_intervention_hook(refusal_direction):
    def intervention_hook(module, input, output):
        s = output[0].shape
        modified_acts = output[0] + refusal_direction.to(output[0].device).unsqueeze(0)
        output = (modified_acts,) + output[1:]
        return output
    return intervention_hook

def evaluate_refusal_direction(args, model, tokenizer, dataset, refusal_direction):
    successes = 0
    for example in tqdm(dataset):

        # Register the intervention hook
        hook = model.model.layers[args.layer].register_forward_hook(
            create_intervention_hook(refusal_direction)
        )

        # Prepare instruction and prompt
        instruction = tokenizer.apply_chat_template(
            [{"role": "user", "content": example["instruction"]}],
            add_generation_prompt=True,
            tokenize=False
        )
        
        # Generate the response
        inputs = tokenizer(instruction, return_tensors="pt").to(model.device)
        input_length = inputs.input_ids.shape[1]
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = outputs[0][input_length:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Remove the intervention hook
        hook.remove()

        evaluation = evaluate_jailbreak(
            completions=[{"prompt": example["instruction"], "response": response, "category": example["category"]}],
            evaluation_path=f"refusal_direction_outputs/{args.model_name}-refusal_direction_evaluation.json")
        
        successes += evaluation["substring_matching_success_rate"]

        print("Instruction: ", instruction)
        print("Response: ", response)
    return (1 - successes / len(dataset))

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

    # Load the datasets
    datasets = {}
    data_dir = "refusal_direction/dataset/splits/"
    for filename in os.listdir(data_dir):
        if filename.endswith('.json'):
            dataset_name = filename.replace('.json', '')
            with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as f:
                datasets[dataset_name] = json.load(f)
    print(f"Loaded {len(datasets)} datasets: {', '.join(datasets.keys())}")

    if args.layer is None:
        # Search over all layers
        n_layers = len(model.model.layers)
        location_performances = []
        location_refusal_directions = []
        location_tuples = []
        
        for position in range(-7, 0, 1):
            args.position = position  # Temporarily set the position
            for layer in tqdm(range(n_layers), desc="Searching layers"):
                print(f"position: {position}, layer: {layer}")

                # Get the model activations on the harmless dataset
                args.layer = layer  # Temporarily set the layer
                harmless_dataset = datasets["harmless_train"][:args.n_refusal_samples]
                harmless_activations = get_mean_activations(args, model, tokenizer, harmless_dataset)

                # Get the model activations on the harmful dataset
                harmful_dataset = datasets["harmful_train"][:args.n_refusal_samples]
                harmful_activations = get_mean_activations(args, model, tokenizer, harmful_dataset)

                # Compute the refusal direction as difference in means
                refusal_direction = torch.mean(harmful_activations, dim=0) - torch.mean(harmless_activations, dim=0)
                location_refusal_directions.append(refusal_direction)

                # Evaluate the refusal direction
                harmless_test = datasets["harmless_test"][:args.n_evaluation_samples]
                performance = evaluate_refusal_direction(args, model, tokenizer, harmless_test, refusal_direction)
                location_performances.append(performance)

                # Memoize parallel location
                location_tuples.append((position, layer))

        # Print the best performing layer
        best_location = np.argmax(location_performances)
        print(f"Best performing layer: {location_tuples[best_location]} with success ratio: {location_performances[best_location]:.3f}")

        # Dump the refusal direction
        with open(f"outputs/{args.model_name}.pkl", "wb") as f:
            #pickle.dump({"best_layer": location_tuples[best_location][1], "best_refusal_direction": location_refusal_directions[best_location], "best_position": location_tuples[best_location][0]}, f)
            pickle.dump({"location_performances": location_performances, "location_refusal_directions": location_refusal_directions, "location_tuples": location_tuples}, f)
    
    else:
        # Original single-layer code
        harmless_dataset = datasets["harmless_train"][:args.n_refusal_samples]
        harmless_activations = get_mean_activations(args, model, tokenizer, harmless_dataset)

        harmful_dataset = datasets["harmful_train"][:args.n_refusal_samples]
        harmful_activations = get_mean_activations(args, model, tokenizer, harmful_dataset)

        refusal_direction = torch.mean(harmful_activations, dim=0) - torch.mean(harmless_activations, dim=0)

        harmless_test = datasets["harmless_test"][:args.n_evaluation_samples]
        test_performance = evaluate_refusal_direction(args, model, tokenizer, harmless_test, refusal_direction)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--position", type=int, default=-1)
    parser.add_argument("--n_refusal_samples", type=int, default=500)
    parser.add_argument("--n_evaluation_samples", type=int, default=40)
    parser.add_argument("--output_dir", type=str, default="outputs_multi_jailbreak")
    args = parser.parse_args()

    main(args)
