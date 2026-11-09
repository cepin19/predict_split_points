#!/bin/bash
# Submit the full TinyStories probing pipeline for one pretrained web-text LM as a chain of
# Slurm jobs (run this on the login node, not via sbatch):
#   fine-tune on TinyStories train (so it ends stories) -> gate: >= GATE_MIN_ENDED of 100 pilot
#   runs end before the cap -> generation (train shards + held-out test) -> merge (drop capped /
#   looped runs) -> per-layer probes for both targets.
# e.g. MODEL=gpt2 ./run_lm_probe_pipeline.sh
set -euo pipefail
cd /lnet/work/people/jon/predict_split_points

# --- Configuration ---
MODEL="${MODEL:?set MODEL, e.g. gpt2}"
NAME="${NAME:-$(basename "$MODEL")}"
FT_DATA="${FT_DATA:-./tinystories/TinyStories-train.head.oneline.txt}"
PROMPTS="${PROMPTS:-./tinystories/TinyStories-valid.oneline.txt}"
FT_DIR="./ft_${NAME}_tinystories"
CAP="${CAP:-1000}"
SAVE_LAYERS="${SAVE_LAYERS:-0,2,4,6,8,10,12}"
N_SHARDS="${N_SHARDS:-10}"
TRAIN_LINES="${TRAIN_LINES:-20000}"
TEST_START="${TEST_START:-20001}"
TEST_END="${TEST_END:-21990}"
TEMPERATURE="${TEMPERATURE:-0.3}"
GATE_MIN_ENDED="${GATE_MIN_ENDED:-0.85}"
EXCLUDE="${EXCLUDE:-dll-8gpu1}"  # node whose prolog hung on 2026-10-20

REUSE_FT="${REUSE_FT:-0}"  # 1: keep an existing $FT_DIR and skip the fine-tune job
for d in $([ "$REUSE_FT" = 1 ] || echo "$FT_DIR") "./${NAME}_ts_gate" "./${NAME}_ts_train" "./${NAME}_ts_test"; do
    [ -e "$d" ] && { echo "Refusing: $d exists" >&2; exit 1; }
done
# generation settings reach the job through the environment (--export splits on commas)
GEN_ENV=(CORPUS="$PROMPTS" MODEL_DIR="$FT_DIR" PROMPT_WORDS=8 MAX_NEW_TOKENS="$CAP" TEMPERATURE="$TEMPERATURE" SAVE_LAYERS="$SAVE_LAYERS")
SB=(sbatch --parsable --exclude="$EXCLUDE" -o "slurm-%x-%j.out")
export MODEL NAME FT_DATA FT_DIR PROMPTS CAP TEST_START GATE_MIN_ENDED N_SHARDS TEMPERATURE  # read by run_lm_stage.sh

if [ "$REUSE_FT" = 1 ]; then
    [ -d "$FT_DIR" ] || { echo "REUSE_FT=1 but $FT_DIR is missing" >&2; exit 1; }
    ft=none; ft_dep=()
else
    ft=$(STAGE=finetune "${SB[@]}" -J "ft-$NAME" -p gpu-ms --gres=gpu:1 --mem=48G --export=ALL run_lm_stage.sh)
    ft_dep=(--dependency=afterok:$ft)
fi
gate=$(STAGE=gate "${SB[@]}" -J "gate-$NAME" "${ft_dep[@]}" -p gpu-ms --gres=gpu:1 --mem=32G --export=ALL run_lm_stage.sh)


gen=()
per=$((TRAIN_LINES / N_SHARDS))
for i in $(seq 0 $((N_SHARDS - 1))); do
    gen+=("$(env "${GEN_ENV[@]}" LINE_START=$((i * per + 1)) LINE_END=$(((i + 1) * per)) OUTPUT=./${NAME}_ts_train_shard$i \
        "${SB[@]}" -J "gen-$NAME" --dependency=afterok:$gate -p gpu-ms --gres=gpu:1 --mem=32G --export=ALL \
        run_elrc_small_inference.sh)")
done
gen+=("$(env "${GEN_ENV[@]}" LINE_START=$TEST_START LINE_END=$TEST_END OUTPUT=./${NAME}_ts_heldout_test \
    "${SB[@]}" -J "gen-$NAME" --dependency=afterok:$gate -p gpu-ms --gres=gpu:1 --mem=32G --export=ALL \
    run_elrc_small_inference.sh)")

merge=$(STAGE=merge "${SB[@]}" -J "merge-$NAME" --dependency=afterok:$(IFS=:; echo "${gen[*]}") -p cpu-troja --gres=none --mem=16G --export=ALL run_lm_stage.sh)

probes=()
for L in ${SAVE_LAYERS//,/ }; do
    for tv in tokens_left:cls tokens_left:squared:log timestep:cls timestep:squared:linear; do
        tgt=${tv%%:*}; v=${tv#*:}; pn=cls; [ "$v" != cls ] && pn=reg_${v/:/_}
        probes+=("$(LAYER=$L TARGET=$tgt VARIANT=$v MAX_RUN_STEPS=$CAP TRAIN_DIR=./${NAME}_ts_train TEST_DIR=./${NAME}_ts_test \
            PREDICTOR_DIR=./saved_predictor_${NAME}_ts_${tgt}_${pn}_layer$L \
            "${SB[@]}" -J "probe-$NAME" --dependency=afterok:$merge -p cpu-troja --gres=none --export=ALL run_elrc_small_train_probe.sh)")
    done
done

echo "$NAME: finetune $ft | gate $gate | generation ${gen[*]} | merge $merge | probes ${probes[*]}"
