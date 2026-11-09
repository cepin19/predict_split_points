#!/bin/bash 


#SBATCH -J trans					  # name of job
#SBATCH -p gpu-troja				  # name of partition or queue (if not specified default partition is used
#SBATCH -D /lnet/work/people/jon/predict_split_points
#SBATCH --gres=gpu:1
#SBATCH  --mem=65G
source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh

eval $(conda shell.bash hook)
conda activate /lnet/work/people/jon/predict_split_points/env
export HOME=/lnet/work/people/jon/llm_agents
python run_inference.py --input-file ../student_models/cssk/cssk.20260211/corp/ELRC-EUR_LEX.cs-sk.sk  --prompt-template "Translate to Czech:\n{line}\n" --model google/gemma-3-4b-it --output-dir ./translations_activations_ELRC-EUR_LEX
