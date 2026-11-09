#!/bin/bash 


#SBATCH -J trans					  # name of job
#SBATCH -p gpu-troja				  # name of partition or queue (if not specified default partition is used
#SBATCH -D /lnet/work/people/jon/predict_split_points
#SBATCH --gres=gpu:1
#SBATCH  --mem=65G
source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh

eval $(conda shell.bash hook)
conda activate /lnet/work/people/jon/predict_split_points/env
export HOME=/lnet/work/people/jon/llm_agents

# Inference mode: translation (corpus lines) or countdown (many prompts / many run dirs).
# Switch by setting INFERENCE_MODE before sbatch, or edit the default below.
INFERENCE_MODE="${INFERENCE_MODE:-translation}"

if [ "$INFERENCE_MODE" = "translation" ]; then
  python run_inference.py \
    --input-file ../student_models/cssk/cssk.20260211/corp/ELRC-EUR_LEX.cs-sk.sk --no-save-scores-softmax \
    --prompt-template "Translate to Czech, only output the single line of translation without any explanations:\n{line}\n" \
    --model google/gemma-3-4b-it \
    --output-dir ./translations_activations_ELRC-EUR_LEX_new
elif [ "$INFERENCE_MODE" = "countdown" ]; then
  # One saved run per (repetition × start value): run0, run1, … under OUTPUT_DIR.
  # Uses prompts "{start} " for each start in 1..COUNTDOWN_MAX (same shape as train_small_transformer).
  MODEL="${COUNTDOWN_MODEL:-./small_lm_out}"
  OUTPUT_DIR="${COUNTDOWN_OUTPUT:-./countdown_activations}"
  COUNTDOWN_MAX="${COUNTDOWN_MAX:-20}"
  REPETITIONS="${COUNTDOWN_REPETITIONS:-1}"
  python run_inference.py \
    --countdown \
    --countdown-max-num "${COUNTDOWN_MAX}" \
    --countdown-repetitions "${REPETITIONS}" \
    --model "${MODEL}" \
    --output-dir "${OUTPUT_DIR}" \
    --max-new-tokens 128 --no-save-scores-softmax \
    --seed 42
  # Optional: fixed starts only, e.g. COUNTDOWN_STARTS="3,7,12" and add to python line:
  #   --countdown-starts "${COUNTDOWN_STARTS}"
  # Optional: shuffle order of starts each repetition:
  #   --countdown-shuffle-starts
else
  echo "Unknown INFERENCE_MODE=${INFERENCE_MODE} (use translation or countdown)" >&2
  exit 1
fi
