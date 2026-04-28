#!/bin/bash -l
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=200g
#SBATCH --tmp=20g
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:h100:4
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=pan00389@umn.edu
#SBATCH --job-name=interactive-gpu

echo "Node ready: $(hostname)"
echo "SSH to it with: ssh $(hostname)"

# Hold the node — Ctrl+C or scancel <jobid> to release
sleep infinity
