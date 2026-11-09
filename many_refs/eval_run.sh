#!/bin/bash
# Full evaluation of an annotate_paper.py run and explorer data file:
#   bash eval_run.sh paper_v5 [sids...]
# 1. explorer data (bubbles -> exact automata of distinct translations, human automata, union)
# 2. near-match statistics (norm / nopunct / skeleton) with the browser engine
# 2b. exact overlaps between sets (overlap.py) and coverage of the WMT11 system outputs (sys_near.js)
# 3. reviewer acceptance per translator before/after QC (eval_ann.py, gpt-6-luna as evaluation reviewer)
# 4. prefix-recall jobs on slurm (prefix_compare.py), merged in when done
# 5. verbatim prompts
# 6. packed copy for the artifact in explorer/pub/ (pack_data.py); publish index.html + engine.js + explorer/pub/data_<tag>*.json
set -euo pipefail
tag=$1; shift
sids=("$@"); [ ${#sids[@]} -eq 0 ] && sids=(7 70 112)
PY=../env_vllm/bin/python
export HF_HOME=/lnet/work/people/jon/llm_agents/.cache/huggingface HF_HUB_OFFLINE=1
out=explorer/data_$tag.json
[ -n "${SKIP_BUILD:-}" ] || $PY build_explorer2.py runs/$tag --out $out   # SKIP_BUILD=1 reuses an existing data file
$PY rebuild_norm.py $out
(cd explorer && DATA=data_$tag.json N=150 node near_stats.js)
$PY overlap.py $out                                   # exact overlaps: LLM pairs, human pair, vs human set
$PY sys_outputs.py $out && (cd explorer && DATA=data_$tag.json SYS=sys_outputs.json node sys_near.js)   # WMT11 system outputs covered
$PY eval_ann.py runs/$tag > logs/eval_ann_$tag.log 2>&1 || true
$PY explorer_prompts.py runs/$tag $out
$PY pack_data.py $out --out explorer/pub            # published copy: packed automata, one file per sentence (16 MB limit per file)
# prefix recall (slurm); merge with: python merge_prefix.py runs/<tag> explorer/data_<tag>.json
for i in "${!sids[@]}"; do :; done
[ -n "${SKIP_PREFIX:-}" ] || sbatch -J pf_$tag --array=0-$((${#sids[@]}-1)) run_prefix.sh runs/$tag --sids "${sids[@]}" || echo "sbatch failed (run prefix_compare.py by hand)"
echo "done: $out"
