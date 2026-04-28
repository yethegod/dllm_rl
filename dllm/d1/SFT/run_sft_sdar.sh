#!/bin/bash -l

export PATH=/users/8/pan00389/miniconda3/envs/d1/bin:$PATH
export CUDA_HOME=/users/8/pan00389/miniconda3/envs/d1/lib/python3.10/site-packages/nvidia/cuda_runtime
export DS_BUILD_OPS=0

cd /users/8/pan00389/dllm/d1/SFT

accelerate launch \
    --config_file ddp_config.yaml \
    --num_processes 4 \
    sft_train_sdar.py \
    --model_name /users/8/pan00389/dllm/models/SDAR-4B-Chat \
    --train_data /users/8/pan00389/dllm/train.jsonl \
    --output_dir /scratch.global/pan00389/sdar-sft \
    --job_name sdar-sft-test \
    --num_epochs 3 \
    --save_steps 500 \
    --batch_size 1 \
    --grad_accum_steps 4 \
    --max_length 1536 \
    --learning_rate 1e-5
