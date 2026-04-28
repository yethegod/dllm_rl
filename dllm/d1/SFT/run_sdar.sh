#!/bin/bash
set -e

# Environment
source /users/8/pan00389/miniconda3/etc/profile.d/conda.sh
conda activate d1

export CUDA_HOME=/common/software/install/manual/cuda/12.1.1
export DS_BUILD_OPS=0
export TRITON_CACHE_DIR=/scratch.global/pan00389/.triton
export WANDB_MODE=disabled
mkdir -p "$TRITON_CACHE_DIR"

cd /users/8/pan00389/dllm/d1/SFT

accelerate launch \
    --config_file ddp_config.yaml \
    --num_processes 1 \
    sft_train_sdar.py \
    --model_name /users/8/pan00389/dllm/models/SDAR-4B-Chat \
    --train_data /users/8/pan00389/dllm/train.jsonl \
    --output_dir /scratch.global/pan00389/sdar-sft \
    --job_name sdar-sft-test \
    --num_epochs 3 \
    --save_steps 500 \
    --batch_size 1 \
    --grad_accum_steps 16 \
    --max_length 1536 \
    --learning_rate 1e-5
