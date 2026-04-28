#!/bin/bash -l
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=200g
#SBATCH --tmp=20g
#SBATCH --partition=preempt-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pan00389@umn.edu
#SBATCH --job-name=sdar-sft
#SBATCH --account=qianwen

export PATH=/users/8/pan00389/miniconda3/envs/d1/bin:$PATH
export CUDA_HOME=/common/software/install/manual/cuda/12.1.1
module load cuda/12.1.1
export DS_BUILD_OPS=0
export WANDB_MODE=disabled
export TRITON_CACHE_DIR=/scratch.global/pan00389/.triton
mkdir -p $TRITON_CACHE_DIR

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
