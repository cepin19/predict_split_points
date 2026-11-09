#!/bin/bash

#SBATCH -J elrc-small-tokens_left-gpt2
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

# --- Configuration ---
CORPUS="${CORPUS:-../student_models/cssk/cssk.20260211/corp/ELRC-EUR_LEX.cs-sk.sk}"
MODEL_DIR="${MODEL_DIR:-./small_lm_elrc_out}"
TRAIN_LINES="${TRAIN_LINES:-2000}"
TEST_LINES="${TEST_LINES:-100}"
PROMPT_WORDS="${PROMPT_WORDS:-3}"   # use first N words of each line as prompt
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-100}"
TRAIN_OUTPUT="${TRAIN_OUTPUT:-./elrc_small_activations_tokens_left_train}"
TEST_OUTPUT="${TEST_OUTPUT:-./elrc_small_activations_tokens_left_test}"
PREDICTOR_DIR="${PREDICTOR_DIR:-./saved_predictor_elrc_small_tokens_left}"
TARGET="tokens_left"
HIDDEN_LAYERS="${HIDDEN_LAYERS:-256,128}"
PREDICTIONS_OUT="${PREDICTIONS_OUT:-${PREDICTOR_DIR}/predictions_${TARGET}_mlp_${HIDDEN_LAYERS//,/_}.csv}"

# Step 1: Train small transformer on ELRC text (skip if model already exists)
echo "=== Step 1: Training small transformer on ELRC text ==="
if [ -d "$MODEL_DIR" ]; then
    echo "Model already exists at $MODEL_DIR, skipping."
else
    # Use full training portion of the corpus as training data
    TRAIN_TMP=$(mktemp)
    head -n "$TRAIN_LINES" "$CORPUS" > "$TRAIN_TMP"
    python train_small_transformer.py \
        --task text \
        --data-path "$TRAIN_TMP" \
        --output-dir "$MODEL_DIR"
    rm -f "$TRAIN_TMP"
fi

# Step 2: Run inference on train subset — prompt with first PROMPT_WORDS words of each line
echo "=== Step 2: Running inference on train subset (lines 1..$TRAIN_LINES) ==="
head -n "$TRAIN_LINES" "$CORPUS" \
    | python3 -c "
import sys
n = int('$PROMPT_WORDS')
for line in sys.stdin:
    words = line.strip().split()
    print(' '.join(words[:n]) if len(words) > n else line.strip())
" \
    | python run_inference.py \
        --input-file /dev/stdin \
        --prompt-template "{line}" \
        --model "$MODEL_DIR" \
        --output-dir "$TRAIN_OUTPUT" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --no-save-scores-softmax

# Step 3: Run inference on test subset
echo "=== Step 3: Running inference on test subset (lines $((TRAIN_LINES+1))..$((TRAIN_LINES+TEST_LINES))) ==="
sed -n "$((TRAIN_LINES+1)),$((TRAIN_LINES+TEST_LINES))p" "$CORPUS" \
    | python3 -c "
import sys
n = int('$PROMPT_WORDS')
for line in sys.stdin:
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

# Step 4: Train predictor on train activations
echo "=== Step 4: Training predictor ==="
python train_predictor.py train \
    --data-dir "$TRAIN_OUTPUT" \
    --model-dir "$PREDICTOR_DIR" \
    --target "$TARGET" \
    --classifier mlp --hidden-layers "$HIDDEN_LAYERS"

# Step 5: Predict on held-out test activations
echo "=== Step 5: Predicting on test set ==="
python train_predictor.py predict \
    --data-dir "$TEST_OUTPUT" \
    --model-dir "$PREDICTOR_DIR" \
    --target "$TARGET" \
    --predictions-out "$PREDICTIONS_OUT"

echo "=== Done. Predictions written to $PREDICTIONS_OUT ==="
