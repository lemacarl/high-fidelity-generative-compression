#!/usr/bin/env bash
# evaluate_reconstruction_low_gan_v9.sh
#
# GPU-side reconstruction-quality evaluation. Runs `python -m tflite.evaluate`
# (restores the TF checkpoint directly, no TFLite conversion) for
# test_image_N (N=1..10) from assets/coffee/paper/ against the
# tflite_low_gan_v9 checkpoint: experiments/tflite_low_gan_v9/final-300000
#
# WHAT v9 CHANGES — nothing in the CLI. v9 runs train_gan_v9.sh with exactly
# v8's flags. The difference is in the model: the rate loss clip was raised
# from -10 to -20 nats in tflite/model/compression_model.py.
#
#   -10 nats caps the rate loss at 14.43 bits per symbol, and clip_by_value
#   zeroes the gradient outside its range. The ANS coder charges 18.48 bits
#   for a |residual|=1 at sigma=0.11 and 17.48 for a tail escape — both above
#   that cap. So the encoder and hyperprior got NO gradient from the only
#   symbols that actually cost bits, sigma parked permanently on the 0.11
#   MIN_SCALE floor, and the adversarial term was free to push latents into
#   expensive territory that training scored as nearly free. On v8, ~4.3% of
#   latents sat in that dead zone and accounted for essentially the entire
#   19.4k-bit coded latent stream.
#
# MEASURED BASELINES ON THIS EXACT IMAGE SET (assets/coffee/paper, n=10),
# coded bpp at each model's own best --scales_min:
#
#     model         LPIPS    PSNR   MS-SSIM   model bpp   coded bpp   ratio
#     low_v2       0.3848   24.86    0.9096      0.1179      0.1565   1.33x
#     low_gan_v7   0.3524   25.24    0.9187      0.1325      0.1556   1.17x
#     low_gan_v8   0.3735   24.11    0.8912      0.1128      0.2162   1.92x
#     low_gan_v9   <- this run
#
# HOW TO JUDGE v9 — the ratio column, not LPIPS.
#   v8 had a genuinely working discriminator (sep positive in 36/36 logged
#   intervals, acc ~0.85) and produced the right qualitative signature: LPIPS
#   better than low_v2 while PSNR and MS-SSIM got worse. It still lost, because
#   it paid +38% coded rate for a 2.9% LPIPS gain. The question for v9 is
#   whether the clip fix closes that gap:
#     coded/model near 1.2x  -> fixed; compare coded bpp against v7's 0.1556
#                               and read LPIPS as meaningful
#     coded/model near 1.9x  -> the clip was not the binding constraint, and
#                               the next lever is --beta (0.15) on the
#                               generator's adversarial weight
#   Note v9's model bpp will read HIGHER than v8's for the same true rate,
#   because the trainer is finally charging full price. That is the fix
#   working, not a regression.
#
# DENSITY_WEIGHTS is passed explicitly and is not optional in practice.
# Checkpoints written before the factorized prior was tracked in the object
# graph contain no prior at all, so without it the rate is computed against a
# randomly-initialised density and BPP is meaningless (~4x too high) while
# PSNR / MS-SSIM stay correct. tflite.evaluate reports which source it used.
#
# Usage:
#   bash evaluate_reconstruction_low_gan_v9.sh
#   bash evaluate_reconstruction_low_gan_v9.sh --csv my_results.csv

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Configurable paths ──────────────────────────────────────────────────────
VARIANT="low_gan_v9"
CKPT="experiments/tflite_low_gan_v9/final-300000"
DENSITY_WEIGHTS="experiments/tflite_low_gan_v9/density_weights.npz"
IMG_DIR="assets/coffee/paper"
OUT_DIR="data/reconstructions/eval/${VARIANT}"
CSV_OUT="evaluate_results_${VARIANT}.csv"
NUM_IMAGES=10
# ────────────────────────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --csv) CSV_OUT="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ ! -f "${CKPT}.index" ]]; then
    echo "ERROR: checkpoint not found: ${CKPT}" >&2
    exit 1
fi

if [[ ! -f "$DENSITY_WEIGHTS" ]]; then
    echo "ERROR: density weights not found: $DENSITY_WEIGHTS" >&2
    echo "Without them BPP is computed against a random prior." >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

# One invocation for all images: the model is built and restored once rather
# than once per image, and metrics come back as CSV instead of being scraped
# out of stdout.
IMAGES=()
for N in $(seq 1 "$NUM_IMAGES"); do
    IMAGES+=("${IMG_DIR}/test_image_${N}.jpg")
done

echo "=== Reconstruction evaluation started at $(date) ==="
echo "Checkpoint: $CKPT"
echo "Results   → $CSV_OUT"
echo ""

python -m tflite.evaluate \
    --checkpoint       "$CKPT" \
    --density_weights  "$DENSITY_WEIGHTS" \
    --images           "${IMAGES[@]}" \
    --out_dir          "$OUT_DIR" \
    --label            "$VARIANT" \
    --csv              "$CSV_OUT"

echo ""
echo "=== Evaluation complete at $(date) ==="
echo "CSV written to: $CSV_OUT"
echo ""
echo "Control row: bash evaluate_reconstruction_low_ft.sh   (no-GAN, matched)"
