#!/bin/bash
set -e

export DS_BUILD_OPS=0
export WANDB_MODE=disabled
export TRITON_CACHE_DIR=/tmp/.triton
mkdir -p "$TRITON_CACHE_DIR"

cd ~/dllm/d1/SFT

accelerate launch \
    --config_file ddp_config_8gpu.yaml \
    --num_processes 8 \
    sft_train_sdar.py \
    --model_name ~/models/SDAR-4B-Chat \
    --train_data ~/dllm/train.jsonl \
    --output_dir ~/dllm/checkpoints/sdar-sft \
    --job_name sdar-sft-lambda \
    --num_epochs 3 \
    --save_steps 250 \
    --batch_size 1 \
    --grad_accum_steps 2 \
    --max_length 1536 \
    --learning_rate 1e-5
