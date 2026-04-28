#!/bin/bash
set -e

# ── Config ─────────────────────────────────────────────────────────────────────
BASE_MODEL=/home/ubuntu/models/SDAR-4B-Chat
CKPT_DIR=/home/ubuntu/checkpoints/sdar-sft-test
OUTPUT_DIR=/home/ubuntu/eval_results
EVAL_SCRIPT=/home/ubuntu/dllm/d1/SFT/eval_sdar.py

# Benchmarks to run (space-separated). Comment out to run all.
BENCHMARKS="math500 aime mmlu gpqa"

# Diffusion settings
GEN_LENGTH=1024
STEPS=128
# ──────────────────────────────────────────────────────────────────────────────

mkdir -p "$OUTPUT_DIR"

CHECKPOINTS=( $(ls -d "$CKPT_DIR"/checkpoint-* | sort -V) )
NUM_CKPTS=${#CHECKPOINTS[@]}
echo "Found $NUM_CKPTS checkpoints, launching one per GPU..."

PIDS=()
for i in "${!CHECKPOINTS[@]}"; do
    CKPT="${CHECKPOINTS[$i]}"
    CKPT_NAME=$(basename "$CKPT")
    LOG="$OUTPUT_DIR/${CKPT_NAME}.log"
    OUT="$OUTPUT_DIR/$CKPT_NAME"
    mkdir -p "$OUT"

    echo "  GPU $i  ->  $CKPT_NAME  (log: $LOG)"
    CUDA_VISIBLE_DEVICES=$i python "$EVAL_SCRIPT" \
        --base_model  "$BASE_MODEL" \
        --checkpoint  "$CKPT" \
        --output_dir  "$OUT" \
        --benchmarks  $BENCHMARKS \
        --gen_length  "$GEN_LENGTH" \
        --steps       "$STEPS" \
        > "$LOG" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "All jobs launched. PIDs: ${PIDS[*]}"
echo "Monitor with:  tail -f $OUTPUT_DIR/checkpoint-*.log"
echo ""

# Wait for all and report exit codes
FAILED=0
for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    CKPT_NAME=$(basename "${CHECKPOINTS[$i]}")
    if wait "$PID"; then
        echo "[OK]   $CKPT_NAME"
    else
        echo "[FAIL] $CKPT_NAME  (exit $?)"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
if [ "$FAILED" -eq 0 ]; then
    echo "All evaluations completed successfully."
    echo "Results in: $OUTPUT_DIR"
else
    echo "$FAILED job(s) failed. Check logs in $OUTPUT_DIR."
    exit 1
fi
