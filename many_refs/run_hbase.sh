#!/bin/bash
#SBATCH -J mr-hbase
#SBATCH -p cpu-ms,cpu-troja
#SBATCH --mem=48G
#SBATCH -c 2
#SBATCH -o logs/hbase_%a.out
set -euo pipefail
export HF_HOME=/lnet/work/people/jon/llm_agents/.cache/huggingface HF_HUB_OFFLINE=1
sids=(7 22 70 109 112)
../env/bin/python human_prefix_baseline.py "${sids[$SLURM_ARRAY_TASK_ID]}"
