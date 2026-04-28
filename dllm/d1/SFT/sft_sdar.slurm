#!/bin/bash -l
#SBATCH --job-name=sdar-sft
#SBATCH --partition=a100-4        # <-- 换成你的 B200 partition 名
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2              # <-- 换成 gpu:b200:2 或对应名称
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --output=/scratch.global/pan00389/sdar-sft-b200-slurm-%j.log
#SBATCH --error=/scratch.global/pan00389/sdar-sft-b200-slurm-%j.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=pstongboa@gmail.com

export PATH=/users/8/pan00389/miniconda3/envs/d1/bin:$PATH
export CUDA_HOME=/users/8/pan00389/miniconda3/envs/d1/lib/python3.10/site-packages/nvidia/cuda_runtime
export DS_BUILD_OPS=0
export TRITON_CACHE_DIR=/scratch.global/pan00389/.triton
export WANDB_MODE=disabled
mkdir -p "$TRITON_CACHE_DIR"

cd /users/8/pan00389/dllm/d1/SFT

accelerate launch \
    --config_file ddp_config_2gpu.yaml \
    --num_processes 2 \
    sft_train_sdar.py \
    --model_name /users/8/pan00389/dllm/models/SDAR-4B-Chat \
    --train_data /users/8/pan00389/dllm/train.jsonl \
    --output_dir /scratch.global/pan00389/sdar-sft-b200 \
    --job_name sdar-sft-b200 \
    --num_epochs 3 \
    --save_steps 500 \
    --batch_size 1 \
    --grad_accum_steps 4 \
    --max_length 1536 \
    --learning_rate 1e-5
