#!/bin/bash

#SBATCH -J elrc_mlp_deep
#SBATCH -p gpu-troja
#SBATCH -D /lnet/work/people/jon/predict_split_points
#SBATCH --gres=gpu:1
#SBATCH --mem=65G
source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh

eval $(conda shell.bash hook)
conda activate /lnet/work/people/jon/predict_split_points/env
export HOME=/lnet/work/people/jon/llm_agents

set -ex
DATA_DIR="${DATA_DIR:-./translations_activations_ELRC-EUR_LEX_train_small/}"
TEST_DIR="${TEST_DIR:-./translations_activations_ELRC-EUR_LEX_test/}"
SAVED_PREDICTORS_DIR="${SAVED_PREDICTORS_DIR:-./predictors_per_layer_ELRC_mlp_large_deep/}"

for layer in 0 1 10 20 34; do
  model_dir="$SAVED_PREDICTORS_DIR/layer_$layer"
  predictions_out="$SAVED_PREDICTORS_DIR/predictions_layer_${layer}.csv"

  echo "=== Layer $layer ==="
  python train_predictor.py train \
    --data-dir "$DATA_DIR" \
    --model-dir "$model_dir" \
    --layer "$layer" \
    --target tokens_left \
    --classifier mlp --hidden-layers 1024,512,512,256

  python train_predictor.py predict \
    --data-dir "$TEST_DIR" \
    --model-dir "$model_dir" \
    --target tokens_left \
    --predictions-out "$predictions_out"
done

echo "Done. Results under $SAVED_PREDICTORS_DIR"
