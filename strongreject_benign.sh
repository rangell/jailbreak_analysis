#!/bin/bash
#
#SBATCH --job-name=benign_strongreject
#SBATCH --output=strongreject_benign.out
#SBATCH -e strongreject_benign.err
#
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=0-24:00:00

singularity exec --nv\
            --overlay /scratch/rca9780/jailbreaks/overlay-15GB-500K-2.ext3:ro \
            /scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif /bin/bash \
            -c 'source /ext3/env.sh; python -u benign_evaluation.py'