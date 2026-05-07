#!/bin/bash -l
#SBATCH --job-name=oc-sdar-sft-math
#SBATCH --partition=preempt-gpu
#SBATCH --nodes=1
#SBATCH --requeue
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128g
#SBATCH --gres=gpu:a40:4
#SBATCH --time=24:00:00
#SBATCH --output=/users/8/pan00389/dllm/SDAR/evaluation/opencompass/outputs/slurm-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=pan00389@umn.edu

export PATH=/users/8/pan00389/miniconda3/envs/opencompass/bin:$PATH
export CUDA_VISIBLE_DEVICES=0,1,2,3

EVAL_DIR=/users/8/pan00389/dllm/SDAR/evaluation/opencompass
cd $EVAL_DIR

echo "=== Starting OpenCompass eval (4× A40) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

python run.py configs/eval_sdar_sft_math.py \
    --mode all \
    --reuse outputs/eval-sdar-sft-math/20260504_190453

echo "=== Eval complete ==="
