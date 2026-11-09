#!/bin/bash

#SBATCH -J countdown-tokens_left-gpt2
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
MODEL_DIR="${MODEL_DIR:-./small_lm_out_new}"
TRAIN_OUTPUT="${TRAIN_OUTPUT:-./countdown_activations_train}"
TEST_OUTPUT="${TEST_OUTPUT:-./countdown_activations_test}"
PREDICTOR_DIR="${PREDICTOR_DIR:-./saved_predictor_countdown_new}"
TRAIN_STARTS="${TRAIN_STARTS:-1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
TEST_STARTS="${TEST_STARTS:-16,17,18,19,20}"
TRAIN_REPS="${TRAIN_REPS:-10}"
TEST_REPS="${TEST_REPS:-3}"
TARGET="${TARGET:-tokens_left}"
HIDDEN_LAYERS="${HIDDEN_LAYERS:-256,128}"
PREDICTIONS_OUT="${PREDICTIONS_OUT:-${PREDICTOR_DIR}/predictions_${TARGET}_mlp_${HIDDEN_LAYERS//,/_}.csv}"

# Step 1: Train small transformer (skip if model already exists)
echo "=== Step 1: Training small transformer ==="
if [ -d "$MODEL_DIR" ]; then
    echo "Model already exists at $MODEL_DIR, skipping."
else
    python train_small_transformer.py --output-dir "$MODEL_DIR"
fi

# Step 2: Extract train activations
echo "=== Step 2: Extracting train activations ==="
python run_inference.py \
    --countdown \
    --countdown-starts "$TRAIN_STARTS" \
    --countdown-repetitions "$TRAIN_REPS" \
    --model "$MODEL_DIR" \
    --output-dir "$TRAIN_OUTPUT" \
    --max-new-tokens 128 --no-save-scores-softmax \
    --seed 42

# Step 3: Extract test activations (held-out start values)
echo "=== Step 3: Extracting test activations ==="
python run_inference.py \
    --countdown \
    --countdown-starts "$TEST_STARTS" \
    --countdown-repetitions "$TEST_REPS" \
    --model "$MODEL_DIR" \
    --output-dir "$TEST_OUTPUT" \
    --max-new-tokens 128 --no-save-scores-softmax \
    --seed 42

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
