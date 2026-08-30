#!/usr/bin/env bash
# resume_gan_matched.sh
#
# Resumes the interrupted matched-rate run in experiments/tflite_low_gan_v6/.
#
# Resume differs from the initial launch in two ways that matter:
#
#   1. --warmstart is NOT passed. The trainer only resumes when it is absent
#      (trainer.py: `if manager.latest_checkpoint and not args.warmstart`);
#      leaving it in would restart from low_v2 at step 0 and discard the work
#      already done.
#
#   2. --prior_weights IS passed. The factorized prior is not written into the
#      checkpoint, so a plain restore would leave it at random initialisation
#      while every other weight and the optimizer state came back correctly —
#      a failure with no visible symptom. The trainer now refuses to resume
#      without it.
#
# --target_bpp must be repeated: rate settings live in the command, not the
# checkpoint, and omitting it would silently fall back to the regime default
# of 0.14 and defeat the point of this run.
#
# Step position is recovered from the checkpoint's save counter, so training
# continues from the last save rather than from zero.
#
# Usage:
#   nohup bash resume_gan_matched.sh > train_matched.log 2>&1 &
#
# If the resume still fails to match optimizer state, add --reset_optimizers
# to the invocation below: weights and discriminator are restored, the Adam
# moments start fresh and rebuild within a few hundred steps.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Must match train_gan_matched.sh exactly ─────────────────────────────────
DATASET="data/coffee/train"
PRIOR_WEIGHTS="experiments/tflite_low_v2/density_weights.npz"
OUT_DIR="experiments/tflite_low_gan_v6/"
REGIME="low"
TARGET_BPP=0.16
N_STEPS=200000
GEN_LR=2e-5
DISC_LR=4e-4
# ────────────────────────────────────────────────────────────────────────────

if [[ ! -f "${OUT_DIR}checkpoint" ]]; then
    echo "ERROR: nothing to resume — no checkpoint in ${OUT_DIR}" >&2
    echo "Start a fresh run with: bash train_gan_matched.sh" >&2
    exit 1
fi

LAST=$(grep -m1 'model_checkpoint_path' "${OUT_DIR}checkpoint" | cut -d'"' -f2)

echo "=== Resuming matched-rate run at $(date) ==="
echo "  checkpoint dir : $OUT_DIR"
echo "  resuming from  : $LAST"
echo "  target steps   : $N_STEPS"
echo "  target bpp     : $TARGET_BPP"
echo "  prior seed     : $PRIOR_WEIGHTS"
echo ""

python -u -m tflite.training.trainer \
    --dataset_path    "$DATASET" \
    --regime          "$REGIME" \
    --target_bpp      "$TARGET_BPP" \
    --model_type      compression_gan \
    --prior_weights   "$PRIOR_WEIGHTS" \
    --gen_lr          "$GEN_LR" \
    --disc_lr         "$DISC_LR" \
    --n_steps         "$N_STEPS" \
    --checkpoint_dir  "$OUT_DIR"

echo ""
echo "=== Resumed run complete at $(date) ==="
