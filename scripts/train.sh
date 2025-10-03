#!/bin/bash
#SBATCH --job-name=pdf2latex_train
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --mail-type=ALL
#SBATCH --mail-user=scottc@cs.toronto.edu
#SBATCH --output=outputs/job_%j.out
#SBATCH --error=outputs/job_%j.err

source ~/miniconda3/etc/profile.d/conda.sh

cd /u/scottc/pdf2latex_llm
conda activate pdf2latex

python pdf2latex/train.py