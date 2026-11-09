#!/bin/bash 


#SBATCH -J tr_pred					  # name of job
#SBATCH -p gpu-troja				  # name of partition or queue (if not specified default partition is used
#SBATCH -D /lnet/work/people/jon/predict_split_points
#SBATCH --gres=gpu:1
#SBATCH  --mem=65G
source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh

eval $(conda shell.bash hook)
conda activate /lnet/work/people/jon/predict_split_points/env
export HOME=/lnet/work/people/jon/llm_agents



# Train a separate predictor for each layer, save each, then run each in predict mode.
# Uses --layer N to train and predict on layer N only. num_layers is discovered from
# the first run's meta.json, or can be set via NUM_LAYERS env var.

set -ex
DATA_DIR="${DATA_DIR:-./translations_activations_ELRC-EUR_LEX_train_small/}"
SAVED_PREDICTORS_DIR="${SAVED_PREDICTORS_DIR:-./predictors_per_layer_ELRC-EUR_LEX_train_small/}"
TEST_FRAC="${TEST_FRAC:-0.2}"
SEED="${SEED:-42}"

#SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#cd "$SCRIPT_DIR"

# Discover number of layers from first run's meta.json (or use NUM_LAYERS env)
if [[ -n "${NUM_LAYERS:-}" ]]; then
  LAYERS=$NUM_LAYERS
else
  FIRST_RUN=$(find "$DATA_DIR" -mindepth 1 -maxdepth 1 -type d -name "run*" | sort -V | head -1)
  if [[ -z "$FIRST_RUN" ]]; then
    echo "No run subdirs found in $DATA_DIR" >&2
    exit 1
  fi
  META="$FIRST_RUN/meta.json"
  if [[ ! -f "$META" ]]; then
    echo "meta.json not found in $FIRST_RUN" >&2
    exit 1
  fi
  LAYERS=$(python3 -c "import json; print(json.load(open('$META'))['num_layers'])")
fi

echo "Training and predicting per layer (0..$((LAYERS - 1)))"
echo "Data dir: $DATA_DIR"
echo "Saved predictors dir: $SAVED_PREDICTORS_DIR"
echo ""

#for layer in $(seq 0 $((LAYERS - 1))); do
	for layer in 1 10 20 34 ; do

  model_dir="$SAVED_PREDICTORS_DIR/layer_$layer"
  predictions_out="$SAVED_PREDICTORS_DIR/predictions_layer_${layer}.csv"

  echo "=== Layer $layer ==="
  echo "  Train..."
  python train_predictor.py train \
    --data-dir "$DATA_DIR" \
    --model-dir "$model_dir" \
    --layer "$layer" \
    --test-frac "$TEST_FRAC" --target tokens_left \
    --seed "$SEED"

  echo "  Predict..."
  python train_predictor.py predict \
    --data-dir ./translations_activations_ELRC-EUR_LEX_test/ \
    --model-dir "$model_dir" --target tokens_left \
    --predictions-out "$predictions_out"
  echo ""
done

echo "Done. Predictors saved under $SAVED_PREDICTORS_DIR/layer_0 .. layer_$((LAYERS - 1))."
echo "Predictions written to $SAVED_PREDICTORS_DIR/predictions_layer_*.csv"
