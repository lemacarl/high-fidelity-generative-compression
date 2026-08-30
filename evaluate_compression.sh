#!/usr/bin/env bash
# run_benchmark.sh
# Runs tflite compression for test_image_N.jpg (N=1..10)
# using both tflite_models_v2_hi and tflite_models_v2_lo,
# measures wall-clock time and computes input vs output file sizes.
# Results are written to compression_benchmark_results.csv.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

INPUT_DIR="data/test_inputs"
OUTPUT_DIR="data/reconstructions"
CSV_OUT="compression_benchmark_results.csv"

MODELS=("hi" "lo", 'med')

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Write CSV header
echo "model_variant,image_n,input_file,output_file,input_size_bytes,output_size_bytes,compression_ratio,elapsed_seconds,status" \
    > "$CSV_OUT"

echo "=== Benchmark started at $(date) ==="
echo "Results will be written to: $CSV_OUT"
echo ""

for X in "${MODELS[@]}"; do
    MODELS_DIR="tflite_models_v2_${X}"
    DENSITY_WEIGHTS="${MODELS_DIR}/density_weights.npz"

    echo "--- Model variant: ${X} (dir: ${MODELS_DIR}) ---"

    # Check that the model dir and density weights exist before looping
    if [[ ! -d "$MODELS_DIR" ]]; then
        echo "  [WARN] Models dir not found: $MODELS_DIR – skipping variant $X"
        for N in $(seq 1 10); do
            echo "${X},${N},data/test_inputs/test_image_${N}.jpg,,,,,SKIPPED_NO_MODEL_DIR" \
                >> "$CSV_OUT"
        done
        continue
    fi

    if [[ ! -f "$DENSITY_WEIGHTS" ]]; then
        echo "  [WARN] density_weights.npz not found: $DENSITY_WEIGHTS – skipping variant $X"
        for N in $(seq 1 10); do
            echo "${X},${N},data/test_inputs/test_image_${N}.jpg,,,,,SKIPPED_NO_DENSITY_WEIGHTS" \
                >> "$CSV_OUT"
        done
        continue
    fi

    for N in $(seq 1 10); do
        INPUT_FILE="${INPUT_DIR}/test_image_${N}.jpg"
        OUTPUT_FILE="${OUTPUT_DIR}/test_image_${N}.tflite.${X}.hfc"

        printf "  [N=%02d X=%s] " "$N" "$X"

        # Check input file exists
        if [[ ! -f "$INPUT_FILE" ]]; then
            echo "SKIP (input not found: $INPUT_FILE)"
            echo "${X},${N},${INPUT_FILE},${OUTPUT_FILE},,,, SKIPPED_NO_INPUT" \
                >> "$CSV_OUT"
            continue
        fi

        INPUT_SIZE=$(stat -c%s "$INPUT_FILE")

        # Time the compression run
        START_TIME=$(date +%s%N)   # nanoseconds

        STATUS="OK"
        if ! python3 -m tflite.inference.compress \
                --compress \
                -i "$INPUT_FILE" \
                -o "$OUTPUT_FILE" \
                --models_dir "$MODELS_DIR" \
                --density_weights "$DENSITY_WEIGHTS" \
                --fp3 \
                2>&1; then
            STATUS="ERROR"
        fi

        END_TIME=$(date +%s%N)
        ELAPSED_NS=$(( END_TIME - START_TIME ))
        # Convert nanoseconds to seconds with 3 decimal places
        ELAPSED_SEC=$(awk "BEGIN { printf \"%.3f\", $ELAPSED_NS / 1000000000 }")

        if [[ "$STATUS" == "OK" && -f "$OUTPUT_FILE" ]]; then
            OUTPUT_SIZE=$(stat -c%s "$OUTPUT_FILE")
            # Compression ratio: input / output  (>1 means output is smaller)
            RATIO=$(awk "BEGIN { printf \"%.4f\", $INPUT_SIZE / $OUTPUT_SIZE }")
            echo "done in ${ELAPSED_SEC}s | input=${INPUT_SIZE}B output=${OUTPUT_SIZE}B ratio=${RATIO}"
        else
            OUTPUT_SIZE=""
            RATIO=""
            echo "FAILED in ${ELAPSED_SEC}s"
            STATUS="ERROR"
        fi

        echo "${X},${N},${INPUT_FILE},${OUTPUT_FILE},${INPUT_SIZE},${OUTPUT_SIZE},${RATIO},${ELAPSED_SEC},${STATUS}" \
            >> "$CSV_OUT"
    done
    echo ""
done

echo "=== Benchmark complete at $(date) ==="
echo "CSV written to: $CSV_OUT"
