import argparse
import itertools
import json
import os
import tempfile


SBATCH_TEMPLATE = """
#!/bin/bash
#
#SBATCH --job-name=__job_name__
#SBATCH --output=__out_path__.out
#SBATCH -e __out_path__.err
#
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:__num_gpus__
#SBATCH --mem=16G
#SBATCH --time=0-0:15:00

singularity exec --nv\
            --overlay /scratch/rca9780/jailbreaks/overlay-15GB-500K.ext3:ro \
            /scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif /bin/bash \
            -c 'source /ext3/env.sh; python -u generate_jailbreak_responses.py --target-model __model_name__ --jailbreak-dataset data/interim/__jailbreak_dataset__.json --output-dir __output_dir__ --max-new-tokens 384'\
"""


if __name__ == "__main__":

    models = [
        "llama3-8b",
        ##"llama3-70b",
        "llama3.1-8b",
        ##"llama3.1-70b",
        "llama3.2-1b",
        #"llama3.2-3b",
        ##"llama3.3-70b",
        ##"gemma-2b",
        #"gemma-7b",
        ##"gemma1.1-2b",
        ##"gemma1.1-7b",
        ##"gemma2-2b",
        ##"gemma2-9b",
        ##"gemma2-27b",
        ##"qwen2.5-0.5b",
        ##"qwen2.5-1.5b",
        ##"qwen2.5-3b",
        ##"qwen2.5-7b",
        ##"qwen2.5-14b",
        ##"qwen2.5-32b"
        ##"outputs_llama-3.2-3B_from_qwen_2.5-7b--checkpoint-40",
        ##"outputs_llama-3.2-3B_from_qwen_2.5-7b--checkpoint-400",
        ##"outputs_gemma2-2b_from_llama3-8b--checkpoint-40",
        ##"outputs_gemma2-2b_from_llama3-8b--checkpoint-400",
        ##"outputs_gemma2-2b_from_qwen2.5-7b--checkpoint-40",
        ##"outputs_gemma2-2b_from_qwen2.5-7b--checkpoint-400",
        ##"outputs_llama3.2-3b_from_gemma2-7b--checkpoint-40",
        ##"outputs_llama3.2-3b_from_gemma2-7b--checkpoint-400",
        ##"outputs_llama3.2-3B_from_llama3-8b--checkpoint-40",
        ##"outputs_llama3.2-3B_from_llama3-8b--checkpoint-400",
        ##"outputs_qwen2.5-3b_from_gemma2-7b--checkpoint-40",
        ##"outputs_qwen2.5-3b_from_gemma2-7b--checkpoint-400",
        ##"gemma2-2b_from_llama3-8b--checkpoint-40",
        ##"gemma2-2b_from_llama3-8b--checkpoint-320",
        ##"gemma2-2b_from_qwen2.5-7b--checkpoint-40",
        ##"gemma2-2b_from_qwen2.5-7b--checkpoint-320",
        #"llama3.2-3b_from_gemma-7b--checkpoint-40",
        #"llama3.2-3b_from_gemma-7b--checkpoint-320",
        ##"llama3.2-3b_from_qwen2.5-7b--checkpoint-40",
        ##"llama3.2-3b_from_qwen2.5-7b--checkpoint-320",
        #"qwen2.5-3b_from_gemma-7b--checkpoint-40",
        #"qwen2.5-3b_from_gemma-7b--checkpoint-320",
        ##"qwen2.5-3b_from_llama3-8b--checkpoint-40",
        ##"qwen2.5-3b_from_llama3-8b--checkpoint-320",
    ]
    jailbreak_datasets = [
        "GCG-meta-llama--Llama-3.2-3B-Instruct",
        #"GCG-scratch--jb9146--git--stanford_alpaca--outputs--distillations--llama3.2-3b_from_gemma-7b--checkpoint-40-----meta-llama--Llama-3.2-3B-Instruct",
        #"GCG-scratch--jb9146--git--stanford_alpaca--outputs--distillations--llama3.2-3b_from_gemma-7b--checkpoint-320-----meta-llama--Llama-3.2-3B-Instruct",
        #"GCG-scratch--jb9146--git--stanford_alpaca--outputs--distillations--qwen2.5-3b_from_gemma-7b--checkpoint-40-----Qwen--Qwen2.5-3B-Instruct",
        #"GCG-scratch--jb9146--git--stanford_alpaca--outputs--distillations--qwen2.5-3b_from_gemma-7b--checkpoint-320-----Qwen--Qwen2.5-3B-Instruct",
    ]
    
    base_output_dir = "/scratch/rca9780/jailbreak_analysis_data/response_shards/"

    for model_name in models:
        for jailbreak_dataset in jailbreak_datasets:
            if "from" in model_name:
                num_gpus = "1"
            else:
                model_size = float(model_name.split("-")[1].replace("b", ""))
                if model_size >= 70:
                    num_gpus = "4"
                elif model_size >= 27:
                    num_gpus = "2"
                else:
                    num_gpus = "1"

            job_name = "{}".format(model_name)
            out_path = "{}/{}/{}".format(base_output_dir, jailbreak_dataset, model_name)
            output_dir = "{}/{}".format(base_output_dir, jailbreak_dataset)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            sbatch_str = SBATCH_TEMPLATE.replace("__job_name__", job_name)
            sbatch_str = sbatch_str.replace("__out_path__", out_path)
            sbatch_str = sbatch_str.replace("__output_dir__", output_dir)
            sbatch_str = sbatch_str.replace("__model_name__", model_name)
            sbatch_str = sbatch_str.replace("__jailbreak_dataset__", jailbreak_dataset)
            sbatch_str = sbatch_str.replace("__num_gpus__", num_gpus)

            print(f"cmd: {model_name}\n")
            with tempfile.NamedTemporaryFile() as f:
                f.write(bytes(sbatch_str.strip(), "utf-8"))
                f.seek(0)
                os.system(f"sbatch {f.name}")