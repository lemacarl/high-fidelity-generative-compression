#!/usr/bin/env bash
# evaluate_reconstruction_low_gan_v8.sh
#
# GPU-side reconstruction-quality evaluation. Runs `python -m tflite.evaluate`
# (restores the TF checkpoint directly, no TFLite conversion) for
# test_image_N (N=1..10) from assets/coffee/paper/ against the
# tflite_low_gan_v8 checkpoint: experiments/tflite_low_gan_v8/final-300000
#
# WHAT v8 IS
#   The first GAN run in this project whose discriminator actually worked.
#   v4-v7 and high_gan all trained with a discriminator pinned at the
#   degenerate constant-output solution, contributing exactly zero adversarial
#   gradient — tflite_low_ft proved it by dropping the adversarial term
#   entirely and reproducing v7 to three decimals. v8 differs from v7 by:
#       --gan_loss_type hinge     (BCE cannot produce the large logits it
#                                  needs under a spectral-norm Lipschitz cap)
#       --disc_ctx_norm layer     (unnormalised latents made the concat ~96%
#                                  context energy, identical in both branches)
#       --n_critic 2              (at 1:1 the generator cancelled the
#                                  discriminator every step; neither got ahead)
#       --disc_lr 1e-4            (was 4e-4)
#       plus discriminator biases restored and one concatenated forward pass
#   Over 300k steps (~100k generator updates) sep stayed positive in every
#   logged interval, acc held ~0.85, D fell from 2.0 to ~1.6.
#
# THE COMPARISON THAT MATTERS — against low_ft, not low_v2.
#   low_ft is the no-GAN control: same data, same warm start, same 100k
#   generator updates, same 0.16 target, differing ONLY in the adversarial
#   term. low_v2 differs in data, steps and rate as well, so a v8-vs-low_v2
#   gap cannot be attributed to the GAN.
#
#   Expect LPIPS BETTER and PSNR WORSE than low_ft. That trade is the
#   rate-distortion-perception tradeoff and is what a working GAN looks like.
#   PSNR going UP instead was the tell that v7 and high_gan were not doing
#   anything adversarial.
#
# IMAGE SET — assets/coffee/paper (10 JPGs), matching the v7 pair.
#   No comparison numbers are quoted here on purpose. The figures this project
#   has been reasoning with — low_v2 0.4276 LPIPS, low_ft 0.4009, v7 0.4000 —
#   were all measured on assets/coffee/test-new (20 PNGs) and do NOT transfer
#   to this set: different images, different count, and JPEG sources rather
#   than PNG. Comparing a paper-set v8 against a test-new low_ft would be
#   measuring the image set, not the model.
#
#   So before reading v8, regenerate the control on THIS set:
#       bash evaluate_reconstruction_low_ft.sh
#   and confirm it too points at assets/coffee/paper. evaluate_results_low_ft.csv
#   was deleted after its last run, so it has to be regenerated regardless.
#
# RATE CAVEAT — CHECK THIS BEFORE BELIEVING ANY LPIPS WIN
#   v8's training bpp averaged 0.1665 over its last 20 logs against a 0.16
#   target, with excursions to 0.188. If v8 codes at a materially higher rate
#   than low_ft, a better LPIPS is bought with bits rather than with the
#   discriminator and the comparison is void. The coded rates from
#   evaluate_compression_low_gan_v8.sh and evaluate_compression_low_ft.sh are
#   the authority here, not the estimates below.
#
# DENSITY_WEIGHTS is passed explicitly and is not optional in practice.
# Checkpoints written before the factorized prior was tracked in the object
# graph contain no prior at all, so without it the rate is computed against a
# randomly-initialised density and BPP is meaningless (~4x too high) while
# PSNR / MS-SSIM stay correct. tflite.evaluate reports which source it used.
#
# Usage:
#   bash evaluate_reconstruction_low_gan_v8.sh
#   bash evaluate_reconstruction_low_gan_v8.sh --csv my_results.csv

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Configurable paths ──────────────────────────────────────────────────────
VARIANT="low_gan_v8"
CKPT="experiments/tflite_low_gan_v8/final-300000"
DENSITY_WEIGHTS="experiments/tflite_low_gan_v8/density_weights.npz"
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
