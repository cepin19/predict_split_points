#!/bin/bash

#SBATCH -J elrc-small-regression
#SBATCH -o slurm-%x-%j.out
#SBATCH -p cpu-troja
#SBATCH -D /lnet/work/people/jon/predict_split_points
#SBATCH --cpus-per-task=8
#SBATCH --mem=65G
source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
eval $(conda shell.bash hook)
conda activate /lnet/work/people/jon/predict_split_points/env
export HOME=/lnet/work/people/jon/llm_agents

set -ex

# Compare classification vs regression probes on the ELRC-small activations, with runs that
# hit the generation cap (no EOS) excluded from both training and the held-out test set.
# Reuses activations from run_elrc_small_*_pipeline.sh and run_elrc_small_heldout_test.sh.

# --- Configuration ---
TEST_DIR="${TEST_DIR:-./elrc_small_activations_heldout_test}"
MAX_RUN_STEPS="${MAX_RUN_STEPS:-250}"   # = MAX_NEW_TOKENS used for generation
HIDDEN_LAYERS="${HIDDEN_LAYERS:-256,128}"

for TARGET in timestep tokens_left; do
    TRAIN_DIR="./elrc_small_activations_${TARGET}_train"
    for KIND in cls reg; do
        PREDICTOR_DIR="./saved_predictor_elrc_small_${TARGET}_${KIND}_cap${MAX_RUN_STEPS}"
        EXTRA=""
        [ "$KIND" = reg ] && EXTRA="--regression"
        echo "=== $TARGET / $KIND ==="
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
    done
done

echo "=== Done ==="
