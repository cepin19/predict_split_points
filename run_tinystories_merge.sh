#!/bin/bash

#SBATCH -J tinystories-merge
#SBATCH -o slurm-%x-%j.out
#SBATCH -p cpu-troja
#SBATCH -D /lnet/work/people/jon/predict_split_points
#SBATCH --mem=16G
source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
eval $(conda shell.bash hook)
conda activate /lnet/work/people/jon/predict_split_points/env
export HOME=/lnet/work/people/jon/llm_agents

set -ex

# Merge TinyStories generation shards into clean train/test dirs (drop capped and looped runs),
# then report generated lengths.
CAP="${CAP:-1000}"
export PYTHONPATH=/lnet/work/people/jon/predict_split_points
python tinystories/merge_runs.py ./tinystories_33m_train "$CAP" ./tinystories_33m_train_shard{0,1,2,3}
python tinystories/merge_runs.py ./tinystories_33m_test "$CAP" ./tinystories_33m_heldout_test
python tinystories/length_report.py "$CAP" ./tinystories_33m_train_shard{0,1,2,3} ./tinystories_33m_heldout_test ./tinystories_33m_train ./tinystories_33m_test
