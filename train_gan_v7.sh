#!/usr/bin/env bash
# train_gan_v7.sh
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
# THIS RUN RE-TESTS THE GAN ON THE REFRESHED COFFEE SET (2291 images,
# replacing the earlier 1799). It keeps target_bpp=0.16 so the result stays
# directly comparable to low_v2 at 0.1626 coded and to v6 at 0.1596 — v6 lost
# on LPIPS at that matched rate (0.4384 vs 0.4276, worse on 17/20).
#
# Caveat worth remembering when reading the result: 2291 images at batch 8
# over ~100k generator updates is ~331 epochs, against ~422 before. The
# discriminator still sees each image hundreds of times, so if v6 failed
# because the training set is too small for adversarial training, this run
# does not change that regime. Fine-tuning on OpenImages (374k images) is the
# experiment that would.
#
# ORIGINAL NOTE — THIS RUN EXISTS TO MATCH BITRATE. v5 coded at 0.1426 bpp against low_v2's
# 0.1626 — 12.3% cheaper but 3.4% worse on LPIPS, which is the one square of
# the comparison where neither model dominates and no conclusion is available.
# Every quality metric improves with rate, so the two are not comparable until
# they sit at the same one.
#
# v5 reached 0.1426 coded from target_bpp=0.14 (ratio 1.019). To land on
# low_v2's 0.1626 the target is 0.1626 / 1.019 = 0.16. Everything else is held
# identical to v5 — same warm start, same LRs, same step budget — so LPIPS
# between v6 and low_v2 becomes a like-for-like comparison.
#
# Also switched to Adam beta_1=0.5 with a two-time-scale rule (the
# discriminator learns faster than the generator), and cut the step budget:
# 200k alternating steps is ~100k generator updates against roughly 1M in v4.
#
# Usage:
#   nohup bash train_gan_v7.sh > train_v7.log 2>&1 &
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

WARMSTART_CKPT="experiments/tflite_low_v2/final-2000000"
PRIOR_WEIGHTS="experiments/tflite_low_v2/density_weights.npz"
OUT_DIR="experiments/tflite_low_gan_v7/"
REGIME="low"
TARGET_BPP=0.16      # v5 used the regime default of 0.14
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

echo "=== GAN fine-tuning (v7) started at $(date) ==="
echo "  dataset     : $DATASET  (${N_IMAGES} images)"
echo "  warm-start  : $WARMSTART_CKPT"
echo "  prior       : $PRIOR_WEIGHTS"
echo "  output      : $OUT_DIR"
echo "  steps       : $N_STEPS  (~$((N_STEPS / 2)) generator updates)"
echo "  target bpp  : $TARGET_BPP  (v5 used 0.14 -> coded 0.1426)"
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
    --target_bpp      "$TARGET_BPP" \
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
