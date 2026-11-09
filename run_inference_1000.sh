#!/usr/bin/env bash
# Run inference 1000 times with random max-new-tokens in [1, 100] and temperature 1.0.
# Each run appends to --output-dir (default: ./activations_out) as run0, run1, ...

set -euo pipefail

N_RUNS=1000
OUTPUT_DIR="${OUTPUT_DIR:-./activations_out}"
MODEL="${MODEL:-gpt2}"
PROMPT="${PROMPT:-A }"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Running $N_RUNS inferences: random length 1-100, temperature 1.0"
echo "Output directory: $OUTPUT_DIR"
echo "Model: $MODEL"
echo ""

for (( i=1; i <= N_RUNS; i++ )); do
  # Random length 1-100 (bash $RANDOM is 0-32767)
  LEN=$(( RANDOM % 100 + 1 ))
  echo "[$i/$N_RUNS] max-new-tokens=$LEN"
  python run_inference.py \
    --model "$MODEL" \
    --prompt "$PROMPT" \
    --max-new-tokens "$LEN" \
    --temperature 1.0 \
    --output-dir "$OUTPUT_DIR"
done

echo ""
echo "Done. $N_RUNS runs saved under $OUTPUT_DIR (run0 .. run$((N_RUNS-1)))."
