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
#SBATCH --mem=16G
#SBATCH --time=0-00:15:00

singularity exec --nv\
            --overlay /scratch/rca9780/jailbreaks/overlay-15GB-500K.ext3:ro \
            /scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif /bin/bash \
            -c 'source /ext3/env.sh; python -u aggregate_response_shards.py --shard-dir __responses_dir__ --target-model __model_name__'\
"""


if __name__ == "__main__":

    models = [
        ##"llama3-8b",
        ##"llama3-70b",
        ##"llama3.1-8b",
        ##"llama3.1-70b",
        ##"llama3.2-1b",
        #"llama3.2-3b",
        ##"llama3.3-70b",
        ##"gemma-2b",
        "gemma-7b",
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
        "llama3.2-3b_from_gemma-7b--checkpoint-320",
        ##"llama3.2-3b_from_qwen2.5-7b--checkpoint-40",
        ##"llama3.2-3b_from_qwen2.5-7b--checkpoint-320",
        ##"qwen2.5-3b_from_gemma-7b--checkpoint-40",
        ##"qwen2.5-3b_from_gemma-7b--checkpoint-320",
        ##"qwen2.5-3b_from_llama3-8b--checkpoint-40",
        ##"qwen2.5-3b_from_llama3-8b--checkpoint-320",
    ]
    responses_dir = "/scratch/rca9780/jailbreak_analysis_data/response_shards"

    # Uncomment for active attacks
    #responses_dir += "/PGD-meta-llama--Llama-3.2-3B-Instruct/"
    #responses_dir += "/PGD-scratch--jb9146--git--stanford_alpaca--outputs--distillations--llama3.2-3b_from_gemma-7b--checkpoint-40-----meta-llama--Llama-3.2-3B-Instruct/"
    responses_dir += "/PGD-scratch--jb9146--git--stanford_alpaca--outputs--distillations--llama3.2-3b_from_gemma-7b--checkpoint-320-----meta-llama--Llama-3.2-3B-Instruct/"

    for model_name in models:
        job_name = "{}".format("j-" + model_name)
        out_path = "{}/{}".format(responses_dir, model_name + "_judged")

        sbatch_str = SBATCH_TEMPLATE.replace("__job_name__", job_name)
        sbatch_str = sbatch_str.replace("__out_path__", out_path)
        sbatch_str = sbatch_str.replace("__responses_dir__", responses_dir)
        sbatch_str = sbatch_str.replace("__model_name__", model_name)
        
        print(f"cmd: {model_name}\n")
        with tempfile.NamedTemporaryFile() as f:
            f.write(bytes(sbatch_str.strip(), "utf-8"))
            f.seek(0)
            os.system(f"sbatch {f.name}")
