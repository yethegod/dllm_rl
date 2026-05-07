#!/bin/bash
source /home/ubuntu/opencompass_env/bin/activate

export CUDA_VISIBLE_DEVICES=0
export COMPASS_DATA_CACHE=/home/ubuntu/uq/opencompass_data
export HF_DATASETS_CACHE=/home/ubuntu/uq/hf_cache/datasets

cd /home/ubuntu/uq/opencompass
python run.py configs/eval_sdar_sft_math.py --mode all \
  2>&1 | tee logs/eval_run_$(date +%Y%m%d_%H%M%S).log
