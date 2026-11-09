#!/bin/bash
# Run eval_run.sh on a Slurm node (rebuild_norm.py on 10^14-derivation sets needs more memory than the head node has):
#   sbatch -J eval_paper_v6 eval_run_slurm.sh paper_v6 7 70 112          (SKIP_BUILD=1 sbatch ... to reuse explorer/data_<tag>.json)
#SBATCH -p cpu-ms
#SBATCH --mem=120G
#SBATCH -c 4
#SBATCH -t 12:00:00
#SBATCH -o logs/slurm_eval_%j.log
cd /lnet/work/people/jon/predict_split_points/many_refs
export PATH="/home/jon/.nvm/versions/node/v20.19.2/bin:$PATH"   # node for near_stats.js / sys_near.js
bash eval_run.sh "$@"
