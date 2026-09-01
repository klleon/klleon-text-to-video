#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 JOB_ID"
    exit
fi

if [ $1 -lt 0 ] || [ $1 -gt 3 ]; then
    echo "JOB_ID must be between 0 and 3"
    exit
fi

JOB_ID=$1
# JOB_ID -> physical GPU (skip GPU 3 = NVIDIA DGX Display)
GPU_MAP=(1 2 3)
GPU_ID=${GPU_MAP[$JOB_ID]}

FILE="cli_demo_quantization.py"
# FILE="cli_demo.py"
MODEL="THUDM/CogVideoX-2b"
PROMPT="prompts/prompt_$JOB_ID.txt"

RUN_COMMAND="CUDA_VISIBLE_DEVICES=$GPU_ID python $FILE --model_path $MODEL --prompt_file $PROMPT --quantization_scheme fp8"
echo "Running command: $RUN_COMMAND"

# Run the inference
eval $RUN_COMMAND