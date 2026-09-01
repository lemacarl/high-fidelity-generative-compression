#!/usr/bin/env bash
# evaluate_compression_low_gan_v8.sh
#
# On-device-style compression benchmark. Runs `python -m tflite.inference.compress`
# (the same entry point used on the Raspberry Pi) against the exported TFLite
# models for the tflite_low_gan_v8 checkpoint, for test_image_N (N=1..10) from
# assets/coffee/paper/.
#
# Unlike evaluate_reconstruction_low_gan_v8.sh (GPU, restores the TF
# checkpoint directly), this script only needs the exported .tflite files +
# density weights and can be copied to and run on the target device as-is.
# It is also the authoritative bitrate: BPP here is measured from the .hfc on
# disk, whereas the reconstruction script reports the entropy model's estimate.
#
# THIS SCRIPT DECIDES WHETHER THE v8 RESULT COUNTS.
#   v8 is the first run whose discriminator actually worked (see the header of
#   the reconstruction script). Its training bpp averaged 0.1665 over the last
#   20 logs against a 0.16 target, with excursions to 0.188 — higher than v7,
#   which coded at 0.1619. If v8 codes materially above low_ft, then any LPIPS
#   improvement was bought with bits rather than with the adversarial term and
#   the comparison is void. Run evaluate_compression_low_ft.sh alongside this
#   and compare the two mean coded rates BEFORE reading anything into LPIPS.
#
#   Both must be run on the SAME image set. This script uses
#   assets/coffee/paper (10 JPGs), matching the v7 pair. The coded rates this
#   project has quoted previously — low_v2 0.1626, low_gan_v7 0.1619 — were
#   measured on assets/coffee/test-new (20 PNGs) and are not comparable to
#   anything produced here; the encoder centre-crops and resizes, so a
#   different source set gives a different rate for the same model.
#
# Two size columns, because they answer different questions:
#   bpp                 bits per coded pixel — the real rate, and the number to
#                       compare against low_ft or against the 0.16 target
#   ratio_vs_input_png  input PNG bytes / .hfc bytes. Inflated, and not a
#                       compression ratio: the encoder centre-crops and resizes
#                       to 256x256 first, so most of the input's pixels are
#                       discarded rather than compressed. Kept for continuity
#                       with earlier CSVs.
#
# elapsed_seconds covers one whole process, TFLite model loading included, so
# it overstates per-image cost on a device that loads once and encodes many.
# Treat it as a relative figure between variants, not as device throughput.
#
# NOTE: tflite_low_gan_v8 must be exported to TFLite first:
#   python -m tflite.conversion.export_tflite \
#       --checkpoint experiments/tflite_low_gan_v8/final-300000 \
#       --out_dir tflite_models_low_gan_v8/
#
# Usage:
#   bash evaluate_compression_low_gan_v8.sh
#   bash evaluate_compression_low_gan_v8.sh --csv my_results.csv

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Configurable paths ──────────────────────────────────────────────────────
VARIANT="low_gan_v8"
# Exported FP32 .tflite models (from `tflite.conversion.export_tflite
#   --checkpoint experiments/tflite_low_gan_v8/final-300000 --out_dir tflite_models_low_gan_v8/`)
MODELS_DIR="tflite_models_low_gan_v8"
DENSITY_WEIGHTS="experiments/tflite_low_gan_v8/density_weights.npz"
IMG_DIR="assets/coffee/paper"
OUTPUT_DIR="data/compressed/${VARIANT}"
CSV_OUT="evaluate_compression_${VARIANT}.csv"
NUM_IMAGES=10
CROP_SIZE=256          # tflite.inference.compress encodes a fixed 256x256 crop
# ────────────────────────────────────────────────────────────────────────────

CODED_PIXELS=$(( CROP_SIZE * CROP_SIZE ))

while [[ $# -gt 0 ]]; do
    case "$1" in
        --csv) CSV_OUT="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "$OUTPUT_DIR"

echo "model_variant,image_n,input_file,output_file,input_size_bytes,output_size_bytes,coded_pixels,bpp,ratio_vs_input_png,elapsed_seconds,status" \
    > "$CSV_OUT"

echo "=== Compression benchmark started at $(date) ==="
echo "Models dir:      $MODELS_DIR"
echo "Density weights: $DENSITY_WEIGHTS"
echo "Results → $CSV_OUT"
echo ""

# Emit one placeholder row per image and exit, so a missing prerequisite is
# visible in the CSV rather than only on the terminal.
skip_all() {
    local reason="$1"
    for N in $(seq 1 "$NUM_IMAGES"); do
        echo "${VARIANT},${N},${IMG_DIR}/test_image_${N}.jpg,,,,${CODED_PIXELS},,,,${reason}" \
            >> "$CSV_OUT"
    done
    exit 1
}

if [[ ! -d "$MODELS_DIR" ]]; then
    echo "[WARN] Models dir not found: $MODELS_DIR – export it first with:"
    echo "  python -m tflite.conversion.export_tflite --checkpoint experiments/tflite_low_gan_v8/final-300000 --out_dir $MODELS_DIR"
    skip_all "SKIPPED_NO_MODEL_DIR"
fi

if [[ ! -f "$DENSITY_WEIGHTS" ]]; then
    echo "[WARN] density_weights.npz not found: $DENSITY_WEIGHTS – skipping"
    skip_all "SKIPPED_NO_DENSITY_WEIGHTS"
fi

TOTAL_BPP=0
N_OK=0

for N in $(seq 1 "$NUM_IMAGES"); do
    INPUT_FILE="${IMG_DIR}/test_image_${N}.jpg"
    OUTPUT_FILE="${OUTPUT_DIR}/test_image_${N}.hfc"

    printf "  [N=%02d] %s " "$N" "$INPUT_FILE"

    if [[ ! -f "$INPUT_FILE" ]]; then
        echo "SKIP (input not found)"
        echo "${VARIANT},${N},${INPUT_FILE},${OUTPUT_FILE},,,${CODED_PIXELS},,,,SKIPPED_NO_INPUT" >> "$CSV_OUT"
        continue
    fi

    INPUT_SIZE=$(stat -c%s "$INPUT_FILE")

    START_TIME=$(date +%s%N)

    STATUS="OK"
    if ! python -m tflite.inference.compress \
            --compress \
            -i "$INPUT_FILE" \
            -o "$OUTPUT_FILE" \
            --models_dir "$MODELS_DIR" \
            --density_weights "$DENSITY_WEIGHTS" \
            >/dev/null 2>&1; then
        STATUS="ERROR"
    fi

    END_TIME=$(date +%s%N)
    ELAPSED_SEC=$(awk "BEGIN { printf \"%.3f\", ($END_TIME - $START_TIME) / 1000000000 }")

    if [[ "$STATUS" == "OK" && -f "$OUTPUT_FILE" ]]; then
        OUTPUT_SIZE=$(stat -c%s "$OUTPUT_FILE")
        BPP=$(awk "BEGIN { printf \"%.4f\", 8 * $OUTPUT_SIZE / $CODED_PIXELS }")
        RATIO=$(awk "BEGIN { printf \"%.4f\", $INPUT_SIZE / $OUTPUT_SIZE }")
        TOTAL_BPP=$(awk "BEGIN { printf \"%.6f\", $TOTAL_BPP + $BPP }")
        N_OK=$(( N_OK + 1 ))
        echo "done in ${ELAPSED_SEC}s | ${OUTPUT_SIZE}B  bpp=${BPP}"
    else
        OUTPUT_SIZE=""
        BPP=""
        RATIO=""
        STATUS="ERROR"
        echo "FAILED in ${ELAPSED_SEC}s"
    fi

    echo "${VARIANT},${N},${INPUT_FILE},${OUTPUT_FILE},${INPUT_SIZE},${OUTPUT_SIZE},${CODED_PIXELS},${BPP},${RATIO},${ELAPSED_SEC},${STATUS}" \
        >> "$CSV_OUT"
done

echo ""
if [[ "$N_OK" -gt 0 ]]; then
    awk "BEGIN { printf \"Mean coded rate over %d images: %.4f bpp\n\", $N_OK, $TOTAL_BPP / $N_OK }"
fi
echo "=== Compression benchmark complete at $(date) ==="
echo "CSV written to: $CSV_OUT"
