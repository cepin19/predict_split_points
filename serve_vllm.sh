#!/bin/bash
#SBATCH -J vllm
#SBATCH -o slurm-%x-%j.out
#SBATCH -D /lnet/work/people/jon/predict_split_points
#SBATCH -p gpu-ms
#SBATCH --gres=gpu:nvidia_rtx_pro_6000:1
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH --exclude=dll-8gpu1

# OpenAI-compatible vLLM server for the many-references pipeline.
#   sbatch serve_vllm.sh                      # Gemma-4-31B-it on one 96 GB RTX Pro 6000
#   MODEL=google/gemma-2-2b-it TP=1 sbatch --gres=gpu:1 --export=ALL serve_vllm.sh
# Once the server is up, vllm_endpoints/<jobid>.json holds base_url, api_key and model
# (readable only by you); clients find a running server through it.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lnet/work/people/jon/predict_split_points/env_vllm
export HF_HOME=/lnet/work/people/jon/llm_agents/.cache/huggingface
export HF_HUB_OFFLINE=1
# no CUDA toolkit on the nodes: FlashInfer cannot JIT its sampling kernels, so use the native sampler
export VLLM_USE_FLASHINFER_SAMPLER=0
# compile caches on the work disk (home has a small quota)
export VLLM_CACHE_ROOT=/lnet/work/people/jon/predict_split_points/.cache/vllm
export TRITON_CACHE_DIR=/lnet/work/people/jon/predict_split_points/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=/lnet/work/people/jon/predict_split_points/.cache/inductor
set -euo pipefail

MODEL="${MODEL:-google/gemma-4-31b-it}"
TP="${TP:-1}"
MAX_LEN="${MAX_LEN:-32768}"
# thinking models: return the reasoning separately from the answer
case "$MODEL" in *gemma-4*) RP="${REASONING_PARSER:-gemma4}" ;; *) RP="${REASONING_PARSER:-}" ;; esac
EXTRA=(); [ -n "$RP" ] && EXTRA=(--reasoning-parser "$RP")
PORT=$((20000 + SLURM_JOB_ID % 20000))
KEY=$(head -c 24 /dev/urandom | base64 | tr -d '/+=')

mkdir -p vllm_endpoints
EP=vllm_endpoints/$SLURM_JOB_ID.json
trap 'rm -f "$EP"' EXIT

vllm serve "$MODEL" --tensor-parallel-size "$TP" --max-model-len "$MAX_LEN" \
    --host 0.0.0.0 --port "$PORT" --api-key "$KEY" --gpu-memory-utilization 0.9 "${EXTRA[@]}" &
PID=$!
until curl -sf -H "Authorization: Bearer $KEY" "http://localhost:$PORT/v1/models" >/dev/null; do
    kill -0 $PID 2>/dev/null || { echo "vllm exited before becoming ready" >&2; exit 1; }
    sleep 10
done
(umask 077; printf '{"base_url": "http://%s:%d/v1", "api_key": "%s", "model": "%s"}\n' \
    "$(hostname -f)" "$PORT" "$KEY" "$MODEL" > "$EP")
echo "ready: $(hostname -f):$PORT model $MODEL"
wait $PID
