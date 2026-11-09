#!/bin/bash
#SBATCH -J mr-prefix
#SBATCH -p cpu-ms,cpu-troja
#SBATCH --mem=64G
#SBATCH -c 2
#SBATCH -o logs/prefix_%x_%a.out
# Usage: sbatch -J pf_TAG --array=0-$((N-1)) run_prefix.sh runs/TAG --sids SID1 SID2 ... [-- extra prefix_compare args]
set -euo pipefail
export HF_HOME=/lnet/work/people/jon/llm_agents/.cache/huggingface HF_HUB_OFFLINE=1
run=$1; shift
[ "$1" == "--sids" ] && shift
sids=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do sids+=("$1"); shift; done
[ "${1:-}" == "--" ] && shift
../env/bin/python prefix_compare.py "$run" "${sids[$SLURM_ARRAY_TASK_ID]}" "$@"
