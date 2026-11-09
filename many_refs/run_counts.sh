#!/bin/bash
#SBATCH -J mr-counts
#SBATCH -p cpu-ms,cpu-troja
#SBATCH --mem=48G
#SBATCH -c 2
#SBATCH -o logs/counts_%a.out
# Usage: sbatch --array=0-49 run_counts.sh   (from many_refs/)
set -euo pipefail
export HF_HOME=/lnet/work/people/jon/llm_agents/.cache/huggingface HF_HUB_OFFLINE=1
sid=$(ls by_sent | sed 's/\.tsv$//' | sort -n | sed -n "$((SLURM_ARRAY_TASK_ID+1))p")
../env/bin/python count_prefixes.py "$sid"
