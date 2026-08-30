#!/usr/bin/env bash
# evaluate_reconstruction_low_ft.sh
#
# GPU-side reconstruction-quality evaluation for the ATTRIBUTION CONTROL run
# (train_ft_control.sh): the same fine-tune as tflite_low_gan_v7 with the
# adversarial term removed. Runs `python -m tflite.evaluate` (restores the TF
# checkpoint directly, no TFLite conversion) for test-image-N (N=1..20) from
# assets/coffee/test-new/.
#
# THE COMPARISON THIS FEEDS, on these same 20 images at matched rate:
#     low_v2       LPIPS 0.4276   PSNR 22.86   MS-SSIM 0.8955   coded 0.1626
#     low_gan_v7   LPIPS 0.4000   PSNR 23.09   MS-SSIM 0.9008   coded 0.1619
#     low_ft       <- this run
#   low_ft landing near 0.400 means the GAN contributed nothing and the gain
#   was domain fine-tuning; near 0.428 means the adversarial term mattered
#   after all, despite D sitting at 2*ln2 for the whole of v7.
#
# IMAGE SET — keep this at assets/coffee/test-new/ (20 PNGs). The numbers
# above were measured there. evaluate_reconstruction_low_gan_v7.sh has since
# been repointed at assets/coffee/paper/ (10 JPGs), so if you want a v7 row
# alongside this one, re-run v7 against test-new rather than comparing across
# the two sets.
#
# Reports PSNR / MS-SSIM / LPIPS and BPP. Side-by-side original|reconstruction
# images are saved under OUT_DIR.
#
# This does NOT exercise the on-device compression path — see
# evaluate_compression_low_ft.sh. The BPP here is the entropy model's estimate,
# not a coded file size; expect it to read below the .hfc rate, which also
# carries the scale-table quantisation overhead and the container header.
#
# DENSITY_WEIGHTS is passed explicitly and is not optional in practice.
# Checkpoints written before the factorized prior was tracked in the object
# graph contain no prior at all, so without it the rate is computed against a
# randomly-initialised density and BPP is meaningless (~4x too high) while
# PSNR / MS-SSIM stay correct. tflite.evaluate reports which source it used.
#
# Usage:
#   bash evaluate_reconstruction_low_ft.sh
#   bash evaluate_reconstruction_low_ft.sh --csv my_results.csv   (override output path)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Configurable paths ──────────────────────────────────────────────────────
VARIANT="low_ft"
CKPT="experiments/tflite_low_ft/final-100000"
DENSITY_WEIGHTS="experiments/tflite_low_ft/density_weights.npz"
IMG_DIR="assets/coffee/test-new"
OUT_DIR="data/reconstructions/eval/${VARIANT}"
CSV_OUT="evaluate_results_${VARIANT}.csv"
NUM_IMAGES=20
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
    IMAGES+=("${IMG_DIR}/test-image-${N}.png")
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
