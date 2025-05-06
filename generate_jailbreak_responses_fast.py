import argparse
import json
import random
import torch
from datasets import load_dataset
from vllm import LLM, SamplingParams

from utils import expand_shortcut_model_name


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

    # load the jailbreak dataset
    jailbreaks_dataset = load_dataset("json", data_files=args.jailbreak_dataset)["train"]
    jailbreaks_dataset = jailbreaks_dataset.map(lambda example: {"jailbreak_prompt_text": example["jailbroken_prompt"][0]["content"]})

    # set the sampling parameters
    sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, max_tokens=args.max_new_tokens, n=args.num_return_sequences)

    # load the model
    model_name_or_path = expand_shortcut_model_name(args.target_model)
    if isinstance(model_name_or_path, tuple):
        llm = LLM(model=model_name_or_path[0], tokenizer=model_name_or_path[1], dtype=torch.bfloat16)
    else:
        llm = LLM(model=model_name_or_path, dtype=torch.bfloat16)

    # generate responses
    convos = [[{"role": "user", "content": s}] for s in jailbreaks_dataset["jailbreak_prompt_text"]]
    outputs = llm.chat(convos, sampling_params)

    # dump responses
    output_fname = f'{args.output_dir}/{args.target_model}-responses-test.json'
    with open(output_fname, 'w') as f:
        for in_data, output in zip(jailbreaks_dataset, outputs):
            in_data.update({"response": [r.text for r in output.outputs]})
            f.write(json.dumps({k: v for k, v in in_data.items()}) + "\n")
