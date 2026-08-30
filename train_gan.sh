#!/usr/bin/env bash
# train_gan.sh
#
# Phase 2 GAN fine-tuning, warm-started from the tflite_low_v2 phase-1
# checkpoint. Re-runs what produced tflite_low_gan_v4, with the three defects
# that invalidated that run fixed:
#
#   1. The discriminator was fed real images paired with a DIFFERENT batch's
#      latents, so it separated the classes by spotting the mismatch rather
#      than by judging image quality. v4's loss curves show it: disc loss hit
#      0.0375 around step 800k, after which generator loss climbed to 2.37 and
#      stayed there.
#   2. The GAN term updated the factorized prior, dragging the rate operating
#      point (training bpp drifted 0.144 -> 0.175).
#   3. The warm-start silently began with a RANDOM entropy model, because the
#      prior was never written to the checkpoint. The trainer now seeds it from
#      density_weights.npz; watch the startup line to confirm.
#
# Also switched to Adam beta_1=0.5 with a two-time-scale rule (the
# discriminator learns faster than the generator), and cut the step budget:
# 200k alternating steps is ~100k generator updates against roughly 1M in v4.
#
# Usage:
#   nohup bash train_gan.sh > train.log 2>&1 &
#   tail -f train.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Configurable ────────────────────────────────────────────────────────────
# get_dataset globs {root}/**/* recursively. test-new/ — the 20 images every
# eval script scores — now lives under assets/coffee/ and is safely outside
# this tree. data/coffee/{test,val} remain in the glob, so 430 held-out images
# are still trained on; set DATASET="data/coffee/train" (1799 images) if those
# splits are meant to stay held out.
DATASET="data/coffee/train"

WARMSTART_CKPT="experiments/tflite_med_v2/final-1500000"
PRIOR_WEIGHTS="experiments/tflite_med_v2/density_weights.npz"
OUT_DIR="experiments/tflite_med_gan/"
REGIME="med"
N_STEPS=200000
GEN_LR=2e-5
DISC_LR=4e-4
# ────────────────────────────────────────────────────────────────────────────

if [[ ! -d "$DATASET" ]]; then
    echo "ERROR: dataset not found: $DATASET" >&2
    exit 1
fi

if [[ ! -f "${WARMSTART_CKPT}.index" ]]; then
    echo "ERROR: warm-start checkpoint not found: ${WARMSTART_CKPT}" >&2
    exit 1
fi

if [[ ! -f "$PRIOR_WEIGHTS" ]]; then
    echo "ERROR: prior weights not found: $PRIOR_WEIGHTS" >&2
    echo "Without them phase 2 starts from a random entropy model." >&2
    exit 1
fi

N_IMAGES=$(find "$DATASET" -type f \
    \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.JPG' \) \
    | wc -l)

echo "=== GAN fine-tuning (v5) started at $(date) ==="
echo "  dataset     : $DATASET  (${N_IMAGES} images)"
echo "  warm-start  : $WARMSTART_CKPT"
echo "  prior       : $PRIOR_WEIGHTS"
echo "  output      : $OUT_DIR"
echo "  steps       : $N_STEPS  (~$((N_STEPS / 2)) generator updates)"
echo "  gen/disc lr : $GEN_LR / $DISC_LR"
echo ""
echo "Confirm the startup line reads 'Factorized prior: absent from checkpoint"
echo "— loaded 12 variables'. A WARNING there means phase 2 is starting from a"
echo "random entropy model and will repeat v4's mistake."
echo ""

# -u disables stdout block buffering. Redirected to a file, python
# otherwise holds print() output in a 4-8KB buffer, so train.log stays
# empty for hours while training runs perfectly well underneath.
python -u -m tflite.training.trainer \
    --dataset_path    "$DATASET" \
    --regime          "$REGIME" \
    --model_type      compression_gan \
    --warmstart \
    --checkpoint      "$WARMSTART_CKPT" \
    --prior_weights   "$PRIOR_WEIGHTS" \
    --gen_lr          "$GEN_LR" \
    --disc_lr         "$DISC_LR" \
    --n_steps         "$N_STEPS" \
    --checkpoint_dir  "$OUT_DIR"

echo ""
echo "=== GAN fine-tuning complete at $(date) ==="
