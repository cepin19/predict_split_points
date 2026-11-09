#!/bin/bash

#SBATCH -J elrc-small-regression-losses
#SBATCH -o slurm-%x-%j.out
#SBATCH -p gpu-ms
#SBATCH -D /lnet/work/people/jon/predict_split_points
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=65G
source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
eval $(conda shell.bash hook)
conda activate /lnet/work/people/jon/predict_split_points/env
export HOME=/lnet/work/people/jon/llm_agents

set -ex

# tokens_left regression probes: loss (squared/absolute/huber) x target scale (linear/log).
# Same data, filter and test set as run_elrc_small_regression.sh, whose
# saved_predictor_elrc_small_tokens_left_reg_cap250 is the squared/linear variant.

# --- Configuration ---
TARGET="tokens_left"
TRAIN_DIR="./elrc_small_activations_${TARGET}_train"
TEST_DIR="${TEST_DIR:-./elrc_small_activations_heldout_test}"
MAX_RUN_STEPS="${MAX_RUN_STEPS:-250}"
HIDDEN_LAYERS="${HIDDEN_LAYERS:-256,128}"
VARIANTS="${VARIANTS:-squared:log absolute:linear absolute:log huber:linear huber:log}"

for V in $VARIANTS; do
    LOSS="${V%%:*}"
    SCALE="${V##*:}"
    EXTRA=""
    [ "$SCALE" = log ] && EXTRA="--log-target"
    PREDICTOR_DIR="./saved_predictor_elrc_small_${TARGET}_reg_${LOSS}_${SCALE}_cap${MAX_RUN_STEPS}"
    echo "=== $TARGET / $LOSS / $SCALE ==="
    python train_predictor.py train \
        --data-dir "$TRAIN_DIR" \
        --model-dir "$PREDICTOR_DIR" \
        --target "$TARGET" \
        --classifier mlp --hidden-layers "$HIDDEN_LAYERS" \
        --max-run-steps "$MAX_RUN_STEPS" \
        --regression --regression-loss "$LOSS" $EXTRA
    python train_predictor.py predict \
        --data-dir "$TEST_DIR" \
        --model-dir "$PREDICTOR_DIR" \
        --predictions-out "${PREDICTOR_DIR}/predictions_heldout.csv"
done

echo "=== Done ==="
