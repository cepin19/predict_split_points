#!/bin/bash

#SBATCH -J elrc-small-lm-train
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

# Train the small ELRC LM once, then check that it actually learned to stop (emit EOS).
# Exits non-zero if too many sanity generations hit the token cap, so dependent
# pipeline jobs (sbatch --dependency=afterok:<this job>) don't run on a broken model.

# --- Configuration ---
CORPUS="${CORPUS:-../student_models/cssk/cssk.20260211/corp/ELRC-EUR_LEX.cs-sk.sk}"
MODEL_DIR="${MODEL_DIR:-./small_lm_elrc_out}"
LM_TRAIN_LINES="${LM_TRAIN_LINES:-20000}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-1e-3}"
MAX_LENGTH="${MAX_LENGTH:-320}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-250}"
SANITY_LINES="${SANITY_LINES:-50}"         # held-out lines after LM_TRAIN_LINES
MAX_CAP_FRAC="${MAX_CAP_FRAC:-0.25}"       # fail if more runs than this hit MAX_NEW_TOKENS
PROMPT_WORDS="${PROMPT_WORDS:-3}"

if [ -e "$MODEL_DIR" ]; then
    echo "Refusing to overwrite existing $MODEL_DIR; move it aside first." >&2
    exit 1
fi

# Step 1: Train
echo "=== Step 1: Training small transformer on lines 1..$LM_TRAIN_LINES ==="
TRAIN_TMP=$(mktemp)
head -n "$LM_TRAIN_LINES" "$CORPUS" > "$TRAIN_TMP"
python train_small_transformer.py \
    --task text \
    --data-path "$TRAIN_TMP" \
    --output-dir "$MODEL_DIR" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --max-length "$MAX_LENGTH"
rm -f "$TRAIN_TMP"

# Step 2: Sanity check — generate on unseen lines and count runs that hit the cap
echo "=== Step 2: Sanity generation on lines $((LM_TRAIN_LINES+1))..$((LM_TRAIN_LINES+SANITY_LINES)) ==="
SANITY_DIR=$(mktemp -d)
sed -n "$((LM_TRAIN_LINES+1)),$((LM_TRAIN_LINES+SANITY_LINES))p" "$CORPUS" \
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
        --output-dir "$SANITY_DIR" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --no-save-scores-softmax

python3 - "$SANITY_DIR" "$MAX_NEW_TOKENS" "$MAX_CAP_FRAC" <<'EOF'
import glob, json, statistics, sys
d, cap, max_frac = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
metas = [json.load(open(f)) for f in glob.glob(f"{d}/run*/meta.json")]
steps = [m["num_steps"] for m in metas]
frac = sum(s >= cap for s in steps) / len(steps)
print(f"Sanity: {len(steps)} runs, median steps {statistics.median(steps)}, hit cap {frac:.2%}")
for m in metas[:3]:
    print(f"  {m['prompt']!r} -> {m['generated_text'][:150]!r}")
sys.exit(0 if frac <= max_frac else 1)
EOF
rm -rf "$SANITY_DIR"
echo "=== Done. Model at $MODEL_DIR ==="
