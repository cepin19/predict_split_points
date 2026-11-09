#!/bin/bash

#SBATCH -J elrc-timestep-gemma3-4b
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
PROMPT_TEMPLATE="${PROMPT_TEMPLATE:-Translate to Czech, only output the single line of translation without any explanations:\n{line}\n}"
MODEL="${MODEL:-google/gemma-3-4b-it}"
TRAIN_LINES="${TRAIN_LINES:-200}"
TEST_LINES="${TEST_LINES:-50}"
TRAIN_OUTPUT="${TRAIN_OUTPUT:-./elrc_activations_timestep_train}"
TEST_OUTPUT="${TEST_OUTPUT:-./elrc_activations_timestep_test}"
PREDICTOR_DIR="${PREDICTOR_DIR:-./saved_predictor_elrc_timestep}"
TARGET="timestep"
HIDDEN_LAYERS="${HIDDEN_LAYERS:-256,128}"
PREDICTIONS_OUT="${PREDICTIONS_OUT:-${PREDICTOR_DIR}/predictions_${TARGET}_mlp_${HIDDEN_LAYERS//,/_}.csv}"

# Step 1: Run inference on train subset (first TRAIN_LINES lines)
echo "=== Step 1: Running inference on train subset (lines 1..$TRAIN_LINES) ==="
head -n "$TRAIN_LINES" "$CORPUS" | python run_inference.py \
    --input-file /dev/stdin \
    --prompt-template "$PROMPT_TEMPLATE" \
    --model "$MODEL" \
    --output-dir "$TRAIN_OUTPUT" \
    --no-save-scores-softmax

# Step 2: Run inference on test subset (next TEST_LINES lines)
echo "=== Step 2: Running inference on test subset (lines $((TRAIN_LINES+1))..$((TRAIN_LINES+TEST_LINES))) ==="
sed -n "$((TRAIN_LINES+1)),$((TRAIN_LINES+TEST_LINES))p" "$CORPUS" | python run_inference.py \
    --input-file /dev/stdin \
    --prompt-template "$PROMPT_TEMPLATE" \
    --model "$MODEL" \
    --output-dir "$TEST_OUTPUT" \
    --no-save-scores-softmax

# Step 3: Train predictor on train activations
echo "=== Step 3: Training predictor ==="
python train_predictor.py train \
    --data-dir "$TRAIN_OUTPUT" \
    --model-dir "$PREDICTOR_DIR" \
    --target "$TARGET" \
    --classifier mlp --hidden-layers "$HIDDEN_LAYERS"

# Step 4: Predict on held-out test activations
echo "=== Step 4: Predicting on test set ==="
python train_predictor.py predict \
    --data-dir "$TEST_OUTPUT" \
    --model-dir "$PREDICTOR_DIR" \
    --target "$TARGET" \
    --predictions-out "$PREDICTIONS_OUT"

echo "=== Done. Predictions written to $PREDICTIONS_OUT ==="
