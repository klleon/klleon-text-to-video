#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 GPU_ID"
    exit
fi

if [ $1 -lt 0 ] || [ $1 -gt 3 ]; then
    echo "GPU_ID must be between 0 and 3"
    exit
fi

GPU_ID=$1
FILE="cli_demo_quantization.py"
# FILE="cli_demo.py"
MODEL="THUDM/CogVideoX-2b"
PROMPT="prompts/prompt_$GPU_ID.txt"

RUN_COMMAND="CUDA_VISIBLE_DEVICES=$GPU_ID python $FILE --model_path $MODEL --prompt_file $PROMPT"
echo "Running command: $RUN_COMMAND"

# Run the inference
eval $RUN_COMMAND