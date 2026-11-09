#!/bin/bash

#SBATCH -J elrc-small-heldout-test
#SBATCH -o slurm-%x-%j.out
#SBATCH -p gpu-ms
#SBATCH -D /lnet/work/people/jon/predict_split_points
#SBATCH --gres=gpu:1
#SBATCH --mem=65G
source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
eval $(conda shell.bash hook)
conda activate /lnet/work/people/jon/predict_split_points/env
export HOME=/lnet/work/people/jon/llm_agents

set -ex

# Evaluate the already-trained ELRC-small predictors on corpus lines the LM never saw
# (after LM_TRAIN_LINES + the sanity-check lines). One inference pass serves both
# targets, since timestep and tokens_left are both derived from the same runs.

# --- Configuration ---
CORPUS="${CORPUS:-../student_models/cssk/cssk.20260211/corp/ELRC-EUR_LEX.cs-sk.sk}"
MODEL_DIR="${MODEL_DIR:-./small_lm_elrc_out}"
TEST_START="${TEST_START:-20051}"
TEST_END="${TEST_END:-$(wc -l < "$CORPUS")}"
PROMPT_WORDS="${PROMPT_WORDS:-3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-250}"
TEST_OUTPUT="${TEST_OUTPUT:-./elrc_small_activations_heldout_test}"
HIDDEN_LAYERS="${HIDDEN_LAYERS:-256,128}"

if [ -e "$TEST_OUTPUT" ]; then
    echo "Refusing to append to existing $TEST_OUTPUT; move it aside first." >&2
    exit 1
fi

# Step 1: Inference on held-out lines
echo "=== Step 1: Running inference on held-out lines $TEST_START..$TEST_END ==="
sed -n "${TEST_START},${TEST_END}p" "$CORPUS" \
    | python3 -c "
import sys
n = int('$PROMPT_WORDS')
for line in sys.stdin:
    if not line.strip():
        continue  # empty prompt crashes generation (corpus ends with a blank line)
    words = line.strip().split()
    print(' '.join(words[:n]) if len(words) > n else line.strip())
" \
    | python run_inference.py \
        --input-file /dev/stdin \
        --prompt-template "{line}" \
        --model "$MODEL_DIR" \
        --output-dir "$TEST_OUTPUT" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --no-save-scores-softmax

# Step 2: Predict with each saved predictor
for TARGET in timestep tokens_left; do
    PREDICTOR_DIR="./saved_predictor_elrc_small_${TARGET}"
    echo "=== Step 2: Predicting $TARGET with $PREDICTOR_DIR ==="
    python train_predictor.py predict \
        --data-dir "$TEST_OUTPUT" \
        --model-dir "$PREDICTOR_DIR" \
        --target "$TARGET" \
        --predictions-out "${PREDICTOR_DIR}/predictions_heldout_${TARGET}_mlp_${HIDDEN_LAYERS//,/_}.csv"
done

echo "=== Done ==="
