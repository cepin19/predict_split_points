#!/bin/bash

#SBATCH -J elrc-small-inference
#SBATCH -o slurm-%x-%j.out
#SBATCH -p gpu-ms
#SBATCH -D /lnet/work/people/jon/predict_split_points
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
eval $(conda shell.bash hook)
conda activate /lnet/work/people/jon/predict_split_points/env
export HOME=/lnet/work/people/jon/llm_agents

set -ex

# Generate with the ELRC-small LM on corpus lines LINE_START..LINE_END and save activations.
# One run directory serves both targets (timestep and tokens_left are derived from the same runs).

# --- Configuration ---
CORPUS="${CORPUS:-../student_models/cssk/cssk.20260211/corp/ELRC-EUR_LEX.cs-sk.sk}"
MODEL_DIR="${MODEL_DIR:-./small_lm_elrc_out}"
LINE_START="${LINE_START:-1}"
LINE_END="${LINE_END:-20000}"
PROMPT_WORDS="${PROMPT_WORDS:-3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-250}"
OUTPUT="${OUTPUT:-./elrc_small_activations_train_20k}"
TEMPERATURE="${TEMPERATURE:-0.3}"
SAVE_LAYERS="${SAVE_LAYERS:-}"  # e.g. 0,2,4; empty = all hidden states

if [ -e "$OUTPUT" ]; then
    echo "Refusing to append to existing $OUTPUT; move it aside first." >&2
    exit 1
fi

echo "=== Running inference on lines $LINE_START..$LINE_END into $OUTPUT ==="
sed -n "${LINE_START},${LINE_END}p" "$CORPUS" \
    | python3 -c "
import sys
n = int('$PROMPT_WORDS')
for line in sys.stdin:
    if not line.strip():
        continue  # empty prompt crashes generation
    words = line.strip().split()
    print(' '.join(words[:n]) if len(words) > n else line.strip())
" \
    | python run_inference.py \
        --input-file /dev/stdin \
        --prompt-template "{line}" \
        --model "$MODEL_DIR" \
        --output-dir "$OUTPUT" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --temperature "$TEMPERATURE" \
        ${SAVE_LAYERS:+--save-layers "$SAVE_LAYERS"} \
        --no-save-scores-softmax

echo "=== Done. $(ls "$OUTPUT" | wc -l) runs in $OUTPUT ==="
