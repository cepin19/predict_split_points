#!/bin/bash

#SBATCH -o slurm-%x-%j.out
#SBATCH -D /lnet/work/people/jon/predict_split_points
source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
eval $(conda shell.bash hook)
conda activate /lnet/work/people/jon/predict_split_points/env
export HOME=/lnet/work/people/jon/llm_agents

set -ex

# One stage of run_lm_probe_pipeline.sh, chosen by STAGE; all parameters come from the environment.
#   finetune: MODEL FT_DATA FT_DIR
#   gate:     NAME PROMPTS FT_DIR CAP TEST_START GATE_MIN_ENDED [TEMPERATURE] (fails if too few pilot runs end)
#   merge:    NAME CAP N_SHARDS
case "$STAGE" in
finetune)
    python finetune_lm.py --model "$MODEL" --data-path "$FT_DATA" --output-dir "$FT_DIR" \
        --epochs 1 --batch-size 16 --lr 5e-5 --max-length 512
    ;;
gate)
    CORPUS="$PROMPTS" MODEL_DIR="$FT_DIR" PROMPT_WORDS=8 MAX_NEW_TOKENS="$CAP" TEMPERATURE="${TEMPERATURE:-0.3}" SAVE_LAYERS=0 \
        LINE_START="$TEST_START" LINE_END=$((TEST_START + 99)) OUTPUT="./${NAME}_ts_gate" \
        bash run_elrc_small_inference.sh
    python3 - "./${NAME}_ts_gate" "$CAP" "$GATE_MIN_ENDED" <<'EOF'
import glob, json, sys
d, cap, need = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
L = [json.load(open(f))["num_steps"] for f in glob.glob(f"{d}/run*/meta.json")]
ended = sum(x < cap for x in L) / len(L)
print(f"gate: {ended:.0%} of {len(L)} pilot runs ended before the cap (need {need:.0%})")
sys.exit(0 if ended >= need else 1)
EOF
    ;;
merge)
    export PYTHONPATH=/lnet/work/people/jon/predict_split_points
    SHARDS=$(seq -s ' ' -f "./${NAME}_ts_train_shard%g" 0 $((N_SHARDS - 1)))
    python tinystories/merge_runs.py "./${NAME}_ts_train" "$CAP" $SHARDS
    python tinystories/merge_runs.py "./${NAME}_ts_test" "$CAP" "./${NAME}_ts_heldout_test"
    python tinystories/length_report.py "$CAP" $SHARDS "./${NAME}_ts_heldout_test" "./${NAME}_ts_train" "./${NAME}_ts_test"
    ;;
*)
    echo "unknown STAGE=$STAGE" >&2
    exit 1
    ;;
esac
