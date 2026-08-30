#!/usr/bin/env bash
# evaluate_reconstruction_high_v2.sh
#
# GPU-side reconstruction-quality evaluation. Runs `python -m tflite.evaluate`
# (restores the TF checkpoint directly, no TFLite conversion) for
# test-image-N (N=1..20) from assets/coffee/test-new/ against the
# tflite_high_v2 checkpoint: experiments/tflite_high_v2/final-1000000
#
# Reports PSNR / MS-SSIM / LPIPS (reconstruction quality) and BPP. Side-by-side
# original|reconstruction images are saved under OUT_DIR.
#
# This does NOT exercise the on-device compression path — see
# evaluate_compression_high_v2.sh for the .hfc-producing device pipeline. The
# BPP here is the entropy model's estimate, not a coded file size; expect it to
# read slightly below the .hfc rate, which also carries the scale-table
# quantisation overhead and the container header.
#
# WHY final-1000000 AND NOT ckpt-1510000
#   experiments/tflite_high_v2/ also holds ckpt-1470000 … ckpt-1510000 from a
#   later run, and `checkpoint` points at ckpt-1510000. Those are NOT usable
#   here: density_weights.npz in that directory is dated with final-1000000
#   (2026-05-18 14:47), so it is the prior for final-1000000, not for the
#   1.51M-step weights. Pairing the 1510000 latents with the 1000000 prior
#   would give a rate that means nothing. final-1000000 is also the exact
#   checkpoint tflite_high_gan warm-started from (see train_high.log), so it is
#   the baseline that makes the high_gan delta attributable.
#
# DENSITY_WEIGHTS is passed explicitly and is not optional in practice.
# Checkpoints written before the factorized prior was tracked in the object
# graph contain no prior at all, so without it the rate is computed against a
# randomly-initialised density and BPP is meaningless (~4x too high) while
# PSNR / MS-SSIM stay correct. tflite.evaluate reports which source it used.
#
# NOTE ON READING THE METRICS
#   LPIPS, PSNR and MS-SSIM are all full-reference distortion measures — they
#   compare the reconstruction to this specific original. LPIPS correlates
#   better with human judgement than the other two, but it is not a measure of
#   realism, and none of the three rewards a generative model for inventing
#   plausible detail. Compare against high_gan only at matched bitrate.
#
# Usage:
#   bash evaluate_reconstruction_high_v2.sh
#   bash evaluate_reconstruction_high_v2.sh --csv my_results.csv   (override output path)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Configurable paths ──────────────────────────────────────────────────────
VARIANT="high_v2"
CKPT="experiments/tflite_high_v2/final-1000000"
DENSITY_WEIGHTS="experiments/tflite_high_v2/density_weights.npz"
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
