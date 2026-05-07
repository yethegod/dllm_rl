#!/bin/bash
export PATH=/users/8/pan00389/miniconda3/envs/opencompass/bin:$PATH
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Point to existing HF cache so already-downloaded datasets are reused
export HF_HOME=/scratch.global/pan00389/hf_cache
export HF_DATASETS_CACHE=/scratch.global/pan00389/hf_cache/datasets
export HUGGINGFACE_HUB_CACHE=/scratch.global/pan00389/hf_cache/hub

# Redirect OpenCompass data cache away from home dir (inode quota risk)
export COMPASS_DATA_CACHE=/scratch.global/pan00389/opencompass_data

cd /users/8/pan00389/dllm/SDAR/evaluation/opencompass

python run.py configs/eval_sdar_sft_math.py --mode all
