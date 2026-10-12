#!/bin/bash
# Run a whole prompt file in one invocation: split the prompts, launch one
# generation process per GPU in parallel, wait for all jobs and print the
# total wall time. Outputs go to ./results/ (same as cli_demo_quantization).
# Each job's log is also streamed to the terminal with a [jobN] prefix
# (tqdm progress bars filtered out); the full log stays in logs/.
#
# Usage: ./2_run_inference_all.sh <prompt-file> [num-jobs]
#   e.g., ./2_run_inference_all.sh prompts_imagenet.txt      # 4 jobs, 1 per GPU
#         ./2_run_inference_all.sh prompts_imagenet.txt 8    # 2 jobs per GPU

set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "Usage: $0 <prompt-file> [num-jobs]"
    exit 1
fi

PROMPT_FILE=$1
NUM_JOBS=${2:-4}
# CUDA device indices of the four A100s. CUDA orders devices FASTEST_FIRST,
# so the A100s are CUDA 0-3 and the DGX Display (unusable) is CUDA 4 --
# this differs from nvidia-smi (PCI) indices. See PROJECT_KNOWLEDGE.md.
GPUS=(0 1 2 3)

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PY_FILE="$SCRIPT_DIR/cli_demo_quantization_v2.py"
MODEL="THUDM/CogVideoX-2b"

if [ ! -f "$PROMPT_FILE" ]; then
    echo "Prompt file not found: $PROMPT_FILE"
    exit 1
fi

SPLIT_DIR="prompts/parallel_split"
LOG_DIR="logs"
# results/ is created up front so the progress counter can always scan it
mkdir -p "$SPLIT_DIR" "$LOG_DIR" results
rm -f "$SPLIT_DIR"/job_*

# Round-robin split so every job gets an equal share (+-1 line, no
# divisibility requirement; `split -n l/N` splits by bytes and can be
# heavily unbalanced)
for ((i = 0; i < NUM_JOBS; i++)); do
    awk -v n="$NUM_JOBS" -v i="$i" 'NR % n == i' "$PROMPT_FILE" > "$SPLIT_DIR/job_$i"
done

TOTAL=$(wc -l < "$PROMPT_FILE")
# Baseline so progress only counts videos created by this run
BASE_DONE=$(find results -maxdepth 1 -name '*.mp4' 2>/dev/null | wc -l)

START=$(date +%s)
PIDS=()
JOB_IDS=()
FOLLOW_PIDS=()
JOB=0
# Stop the log followers if this script is interrupted. Note: the generation
# jobs themselves keep running (background jobs ignore SIGINT).
trap 'kill "${FOLLOW_PIDS[@]}" 2>/dev/null || true' EXIT
for part in "$SPLIT_DIR"/job_*; do
    if [ ! -s "$part" ]; then
        continue
    fi
    GPU=${GPUS[$((JOB % ${#GPUS[@]}))]}
    LOG="$LOG_DIR/gen_job_${JOB}_gpu${GPU}.log"
    echo "Job $JOB -> CUDA device $GPU ($(wc -l < "$part") prompts, log: $LOG)"
    # -u: unbuffered stdout so "Prompt: ..." lines reach the log immediately
    CUDA_VISIBLE_DEVICES=$GPU python -u "$PY_FILE" \
        --model_path "$MODEL" \
        --prompt_file "$part" \
        --quantization_scheme fp8 > "$LOG" 2>&1 &
    PIDS+=($!)
    JOB_IDS+=("$JOB")
    # Mirror the log to the terminal: split tqdm's \r redraws into lines,
    # strip ANSI cursor codes and the bar prefix before "Prompt:", then drop
    # progress-bar and blank lines. tail exits once the job process exits.
    tail -n +1 -f --pid="${PIDS[-1]}" "$LOG" \
        | stdbuf -oL tr '\r' '\n' \
        | sed -u -e 's/\x1b\[[0-9;]*[A-Za-z]//g' -e 's/^.*\(Prompt: \)/\1/' \
        | grep --line-buffered -v -E '[0-9]+%\||^[[:space:]]*$' \
        | sed -u "s/^/[job$JOB] /" &
    FOLLOW_PIDS+=($!)
    JOB=$((JOB + 1))
done

# Live progress: refresh one status line every 15s until all jobs exit
# (printed as a normal line every 60s, plus once at the end, so it does not
# clobber the streamed job logs)
TICK=0
while :; do
    RUNNING=0
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            RUNNING=$((RUNNING + 1))
        fi
    done
    NOW_DONE=$(( $(find results -maxdepth 1 -name '*.mp4' 2>/dev/null | wc -l) - BASE_DONE ))
    ELAPSED=$(( $(date +%s) - START ))
    ETA_STR="--:--:--"
    if [ "$NOW_DONE" -gt 0 ]; then
        ETA=$(( (TOTAL - NOW_DONE) * ELAPSED / NOW_DONE ))
        ETA_STR=$(printf '%02d:%02d:%02d' $((ETA / 3600)) $((ETA % 3600 / 60)) $((ETA % 60)))
    fi
    if [ $((TICK % 4)) -eq 0 ] || [ "$RUNNING" -eq 0 ]; then
        printf '[elapsed %02d:%02d:%02d] %d/%d videos, %d/%d jobs running, ETA %s\n' \
            $((ELAPSED / 3600)) $((ELAPSED % 3600 / 60)) $((ELAPSED % 60)) \
            "$NOW_DONE" "$TOTAL" "$RUNNING" "${#PIDS[@]}" "$ETA_STR"
    fi
    if [ "$RUNNING" -eq 0 ]; then
        break
    fi
    TICK=$((TICK + 1))
    sleep 15
done
# Let the log followers flush the jobs' last lines before the summary
wait "${FOLLOW_PIDS[@]}" 2>/dev/null || true
echo

FAILED=0
for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
        echo "Job ${JOB_IDS[$i]} FAILED (see $LOG_DIR/gen_job_${JOB_IDS[$i]}_*.log)"
        FAILED=1
    fi
done

ELAPSED=$(( $(date +%s) - START ))
printf 'Total generation time: %02d:%02d:%02d (%d prompts, %d jobs)\n' \
    $((ELAPSED / 3600)) $((ELAPSED % 3600 / 60)) $((ELAPSED % 60)) \
    "$TOTAL" "$NUM_JOBS"

exit $FAILED
