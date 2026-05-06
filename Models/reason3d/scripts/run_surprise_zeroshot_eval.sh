#!/usr/bin/env bash
# Zero-shot Reason3D eval on Surprise val JSON; skips annotations with no GT instance in .pth.
# Requires pointgroup_ops: run scripts/build_pointgroup_ops.sh once per conda env.
#
# Variants (set CFG= to match your checkpoint):
#   bare:        lavis/projects/reason3d/val/reason3d_surprise_zeroshot.yaml (default)
#   geo:         lavis/projects/reason3d/val/reason3d_surprise_zeroshot_geo.yaml
#   chain:       lavis/projects/reason3d/val/reason3d_surprise_zeroshot_chain.yaml
#   chainv3:     lavis/projects/reason3d/val/reason3d_surprise_zeroshot_chainv3.yaml
#                (uses task 3d_refer_seg_v3, builder 3d_refer_chainv3, model arch reason3d_t5_chainv3;
#                 reports per-instance hit@0.25 / hit@0.50 / meanMaxIoU)
#   chainv3_cot: lavis/projects/reason3d/val/reason3d_surprise_zeroshot_chainv3_cot.yaml
#                (model arch reason3d_t5_chainv3_cot, builder 3d_refer_chainv3_cot;
#                 same task as chainv3, but the two-pass predict_seg also surfaces
#                 the pass-1 intermediate mask M_1 so the eval task can persist it
#                 alongside pred/gt -- see REASON3D_SAVE_EVAL_MASKS below)
#
# JSONL without mask .npz (default when saving preds from this script):
#   REASON3D_SAVE_PREDS=1   # qualitative/predictions.jsonl
#   REASON3D_SAVE_EVAL_MASKS=1   # also write qualitative/masks/*.npz (large on full val)
#                                # for chainv3_cot, each .npz row from a CoT two-pass
#                                # decode also contains pred_pmask_intermediate
#                                # (see ``intermediate_in_npz`` in predictions.jsonl)
#   REASON3D_SAVE_PREDS=0   # force no JSONL even if YAML has true
#
# Multi-GPU eval (shard test set; requires torchrun + NCCL):
#   NPROC=4 MASTER_ADDR=127.0.0.1 MASTER_PORT=29511 bash scripts/run_surprise_zeroshot_eval.sh
#
# Point cloud .pth dir (default YAML uses pth_rel_subdir=processed under points.storage):
#   REASON3D_PTH_SUBDIR=processed_surprise_full_pth
#   REASON3D_PTS_ROOT=/nfs-stor/lan.wei/data/scannetpp   # omit if same as YAML
# filter_missing_gt_in_pth is forced ON here (drops QA rows with no object_id in .pth). Opt out:
#   REASON3D_FILTER_MISSING_GT_IN_PTH=0
#
# Resume after crash (reuse the same output job folder name under lavis/<output_dir>/):
#   REASON3D_EVAL_RESUME=1 REASON3D_EVAL_JOB_ID=<timestamp folder>
#   Same CFG and run.output_dir in YAML as the partial run. Use REASON3D_SAVE_PREDS=1 (or YAML save_eval_predictions true).
#   With multi-GPU + DistributedSampler, prefer resuming single-GPU or verify sampler length matches filtered dataset.
#
# Auto-resume on crash (e.g. CUDA OOM): default ON; opt out with REASON3D_AUTO_RESUME=0.
#   - Pre-generates REASON3D_EVAL_JOB_ID so all retries write into the same lavis/<output_dir>/<id>/ folder.
#   - Forces run.save_eval_predictions=true + run.eval_resume_predictions=true so each retry appends to
#     qualitative/predictions.jsonl and skips already-completed (scene_id, ann_id) pairs.
#   - Aborts after REASON3D_MAX_RETRIES (default 10) or if no new predictions were written between attempts
#     (to avoid looping on a sample that always crashes; see logs for the offending row).
#   - Ctrl-C (exit 130 / 128+SIGINT) is treated as user cancel and is not retried.
#   - Sleeps REASON3D_RESUME_BACKOFF_SECONDS (default 5) between attempts to let GPU memory settle.
set -euo pipefail

REASON3D="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REASON3D"
export PYTHONPATH="${REASON3D}:${PYTHONPATH:-}"

# srun/Slurm shells often lack conda on PATH (same as run_surprise_zeroshot_eval_small.sh).
# shellcheck source=/dev/null
. "${REASON3D}/scripts/conda_init_reason3d.sh"

CFG="${CFG:-lavis/projects/reason3d/val/reason3d_surprise_zeroshot.yaml}"
CKPT="${REASON3D_CKPT:-}"

if [[ -z "$CKPT" ]]; then
  echo "Set REASON3D_CKPT to the full Reason3D checkpoint (.pth), e.g.:" >&2
  echo "  REASON3D_CKPT=/path/to/reason3d.pth bash $0" >&2
  exit 1
fi
if [[ ! -f "$CKPT" ]]; then
  echo "REASON3D_CKPT is not a readable file: $CKPT" >&2
  echo "Use an absolute path if your cwd is not the directory that contains the checkpoint." >&2
  exit 1
fi

_append_save_pred_opts() {
  if [[ "${REASON3D_SAVE_PREDS:-}" == "1" ]]; then
    OPTS+=( "run.save_eval_predictions=true" )
    if [[ "${REASON3D_SAVE_EVAL_MASKS:-0}" == "1" ]]; then
      OPTS+=( "run.save_eval_prediction_masks=true" )
    else
      OPTS+=( "run.save_eval_prediction_masks=false" )
    fi
  elif [[ "${REASON3D_SAVE_PREDS:-}" == "0" ]]; then
    OPTS+=( "run.save_eval_predictions=false" )
  fi
}

OPTS=( "model.reason3d_checkpoint=${CKPT}" )
_append_save_pred_opts
# Pick the dataset-config key that matches the chosen CFG. Match the more
# specific *_cot YAML before the plain chainv3 one so the case fall-through
# resolves to the CoT builder when both patterns exist.
_dataset_key_for_cfg() {
  case "$1" in
    *zeroshot_geo.yaml) echo "3d_refer_geo" ;;
    *zeroshot_chain.yaml|*small_chain.yaml) echo "3d_refer_chain" ;;
    *zeroshot_chainv3_cot.yaml) echo "3d_refer_chainv3_cot" ;;
    *zeroshot_chainv3.yaml) echo "3d_refer_chainv3" ;;
    *) echo "3d_refer" ;;
  esac
}
DKEY="$(_dataset_key_for_cfg "$CFG")"

if [[ -n "${REASON3D_PTH_SUBDIR:-}" ]]; then
  OPTS+=( "datasets.${DKEY}.dataset_init.pth_rel_subdir=${REASON3D_PTH_SUBDIR}" )
fi
if [[ -n "${REASON3D_PTS_ROOT:-}" ]]; then
  OPTS+=( "datasets.${DKEY}.build_info.points.storage=${REASON3D_PTS_ROOT}" )
fi
if [[ "${REASON3D_FILTER_MISSING_GT_IN_PTH:-1}" == "0" ]]; then
  OPTS+=( "datasets.${DKEY}.dataset_init.filter_missing_gt_in_pth=false" )
else
  OPTS+=( "datasets.${DKEY}.dataset_init.filter_missing_gt_in_pth=true" )
fi
if [[ "${REASON3D_EVAL_RESUME:-0}" == "1" ]]; then
  if [[ -z "${REASON3D_EVAL_JOB_ID:-}" ]]; then
    echo "REASON3D_EVAL_RESUME=1 requires REASON3D_EVAL_JOB_ID to the interrupted run id (see lavis/output/<cfg>/<id>/)." >&2
    exit 1
  fi
  OPTS+=( "run.eval_resume_predictions=true" "run.save_eval_predictions=true" )
  if [[ "${REASON3D_SAVE_EVAL_MASKS:-0}" == "1" ]]; then
    OPTS+=( "run.save_eval_prediction_masks=true" )
  else
    OPTS+=( "run.save_eval_prediction_masks=false" )
  fi
fi

AUTO_RESUME="${REASON3D_AUTO_RESUME:-1}"
MAX_RETRIES="${REASON3D_MAX_RETRIES:-10}"
RESUME_BACKOFF="${REASON3D_RESUME_BACKOFF_SECONDS:-5}"

if [[ "$AUTO_RESUME" == "1" ]]; then
  if [[ "${REASON3D_SAVE_PREDS:-}" == "0" ]]; then
    echo "[auto-resume] WARNING: REASON3D_SAVE_PREDS=0 conflicts with auto-resume; forcing predictions on." >&2
  fi
  # Pre-bake the job_id so the very first attempt and every retry land in the same output folder.
  if [[ -z "${REASON3D_EVAL_JOB_ID:-}" ]]; then
    export REASON3D_EVAL_JOB_ID="$(date +%Y%m%d%H%M%S)"
  fi
  # save_eval_predictions=true is mandatory (resume needs the JSONL); eval_resume_predictions=true is
  # a no-op on the first attempt (refer_seg_task wipes qualitative/ when the JSONL is empty/missing).
  OPTS+=( "run.save_eval_predictions=true" "run.eval_resume_predictions=true" )
  if [[ "${REASON3D_SAVE_EVAL_MASKS:-0}" == "1" ]]; then
    OPTS+=( "run.save_eval_prediction_masks=true" )
  else
    OPTS+=( "run.save_eval_prediction_masks=false" )
  fi
fi

# Resolve run.output_dir from the YAML so we can watch progress in qualitative/predictions.jsonl.
# Tolerates quoted/unquoted values and trailing comments; takes the first match.
_extract_run_output_dir() {
  grep -E '^[[:space:]]*output_dir:' "$1" \
    | head -n1 \
    | sed -E 's/^[[:space:]]*output_dir:[[:space:]]*//; s/[[:space:]]*#.*$//; s/^"//; s/"$//; s/^'"'"'//; s/'"'"'$//'
}
RUN_OUT_REL=""
if [[ -n "${REASON3D_EVAL_JOB_ID:-}" ]]; then
  RUN_OUT_REL="$(_extract_run_output_dir "$CFG" || true)"
fi
PRED_JSONL=""
if [[ -n "$RUN_OUT_REL" && -n "${REASON3D_EVAL_JOB_ID:-}" ]]; then
  PRED_JSONL="${REASON3D}/lavis/${RUN_OUT_REL}/${REASON3D_EVAL_JOB_ID}/qualitative/predictions.jsonl"
fi

_jsonl_lines() {
  if [[ -n "$PRED_JSONL" && -f "$PRED_JSONL" ]]; then
    wc -l < "$PRED_JSONL" 2>/dev/null | tr -d ' '
  else
    echo 0
  fi
}

NPROC="${NPROC:-1}"
# evaluate.py + init_distributed_mode: YAML distributed:false exits before reading RANK; must override.
if [[ "$NPROC" -gt 1 ]]; then
  OPTS+=( "run.distributed=true" "run.use_dist_eval_sampler=true" )
  if [[ "$AUTO_RESUME" == "1" ]]; then
    echo "[auto-resume] NOTE: NPROC=$NPROC. apply_eval_resume_skip changes dataset length, which can" >&2
    echo "[auto-resume]       break DistributedSampler sharding on resume. Prefer NPROC=1, or set" >&2
    echo "[auto-resume]       REASON3D_AUTO_RESUME=0 and resume manually." >&2
  fi
fi

_run_eval() {
  if [[ "$NPROC" -gt 1 ]]; then
    torchrun --nproc_per_node="$NPROC" evaluate.py --cfg-path "$CFG" --options "${OPTS[@]}"
  else
    python evaluate.py --cfg-path "$CFG" --options "${OPTS[@]}"
  fi
}

if [[ "$AUTO_RESUME" != "1" ]]; then
  exec_cmd_rc=0
  _run_eval || exec_cmd_rc=$?
  exit "$exec_cmd_rc"
fi

attempt=0
prev_lines="$(_jsonl_lines)"
echo "[auto-resume] job_id=${REASON3D_EVAL_JOB_ID}  predictions=${PRED_JSONL:-<unknown>}"
echo "[auto-resume] starting at ${prev_lines} completed predictions; max_retries=${MAX_RETRIES}, backoff=${RESUME_BACKOFF}s"
while :; do
  attempt=$((attempt + 1))
  echo "[auto-resume] === attempt ${attempt}/${MAX_RETRIES} ==="
  rc=0
  _run_eval || rc=$?
  if [[ $rc -eq 0 ]]; then
    echo "[auto-resume] eval finished successfully on attempt ${attempt}."
    exit 0
  fi
  # 130 = SIGINT (Ctrl-C). 131 = SIGQUIT. Don't retry on user interrupts.
  if [[ $rc -eq 130 || $rc -eq 131 ]]; then
    echo "[auto-resume] interrupted by user (exit ${rc}); not retrying." >&2
    exit "$rc"
  fi
  if [[ $attempt -ge $MAX_RETRIES ]]; then
    echo "[auto-resume] reached max retries (${MAX_RETRIES}); giving up. last exit ${rc}." >&2
    exit "$rc"
  fi
  cur_lines="$(_jsonl_lines)"
  : "${cur_lines:=0}"
  if [[ "$cur_lines" -le "$prev_lines" ]]; then
    echo "[auto-resume] no new predictions written since previous attempt (lines stuck at ${cur_lines})." >&2
    echo "[auto-resume] aborting to avoid looping on a deterministically-failing sample. last exit ${rc}." >&2
    exit "$rc"
  fi
  echo "[auto-resume] crashed with exit ${rc}; ${cur_lines} predictions saved (was ${prev_lines}). Resuming in ${RESUME_BACKOFF}s..."
  prev_lines="$cur_lines"
  sleep "$RESUME_BACKOFF"
done
