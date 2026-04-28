#!/bin/bash
export CUDA_HOME=/common/software/install/manual/cuda/12.1.1
export DS_BUILD_OPS=0
export TRITON_CACHE_DIR=/scratch.global/pan00389/.triton

python sft_train_sdar.py \
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
