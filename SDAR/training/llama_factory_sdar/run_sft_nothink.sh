#!/bin/bash
set -e

cd /home/ubuntu/agent/SDAR/training/llama_factory_sdar

export PYTHONPATH=src
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false

torchrun \
  --nproc_per_node=8 \
  --master_port=29500 \
  src/train.py \
  examples/train_full_sdar/sdar_4b/sdar_4b_nothink_full.yaml \
  2>&1 | tee /lambda/nfs/agent/checkpoints/sdar_4b_nothink/train.log
