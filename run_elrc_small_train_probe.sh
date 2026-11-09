#!/bin/bash

#SBATCH -J elrc-small-train-probe
#SBATCH -o slurm-%x-%j.out
#SBATCH -p gpu-ms
#SBATCH -D /lnet/work/people/jon/predict_split_points
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
eval $(conda shell.bash hook)
conda activate /lnet/work/people/jon/predict_split_points/env
export HOME=/lnet/work/people/jon/llm_agents

set -ex

# Train one probe variant on TRAIN_DIR and predict on the held-out test set.
# VARIANT: cls, or <loss>:<scale> for a regressor, e.g. squared:log, absolute:linear.
# e.g. VARIANT=squared:log sbatch run_elrc_small_train_probe.sh

# --- Configuration ---
TARGET="${TARGET:-tokens_left}"
VARIANT="${VARIANT:-squared:log}"
TRAIN_DIR="${TRAIN_DIR:-./elrc_small_activations_train_20k}"
TEST_DIR="${TEST_DIR:-./elrc_small_activations_heldout_test}"
MAX_RUN_STEPS="${MAX_RUN_STEPS:-250}"
HIDDEN_LAYERS="${HIDDEN_LAYERS:-256,128}"
TAG="${TAG:-20k}"
LAYER="${LAYER:-}"  # hidden-state index (0 = embeddings); empty = concat all layers

if [ "$VARIANT" = cls ]; then
    EXTRA=""
    NAME="cls"
else
    LOSS="${VARIANT%%:*}"
    SCALE="${VARIANT##*:}"
    EXTRA="--regression --regression-loss $LOSS"
    [ "$SCALE" = log ] && EXTRA="$EXTRA --log-target"
    NAME="reg_${LOSS}_${SCALE}"
fi
if [ -n "$LAYER" ]; then
    EXTRA="$EXTRA --layer $LAYER"
    TAG="${TAG}_layer${LAYER}"
fi
PREDICTOR_DIR="${PREDICTOR_DIR:-./saved_predictor_elrc_small_${TARGET}_${NAME}_cap${MAX_RUN_STEPS}_${TAG}}"

echo "=== $TARGET / $VARIANT on $TRAIN_DIR -> $PREDICTOR_DIR ==="
python train_predictor.py train \
    --data-dir "$TRAIN_DIR" \
    --model-dir "$PREDICTOR_DIR" \
    --target "$TARGET" \
    --classifier mlp --hidden-layers "$HIDDEN_LAYERS" \
    --max-run-steps "$MAX_RUN_STEPS" $EXTRA
python train_predictor.py predict \
    --data-dir "$TEST_DIR" \
    --model-dir "$PREDICTOR_DIR" \
    --predictions-out "${PREDICTOR_DIR}/predictions_heldout.csv"

echo "=== Done ==="
