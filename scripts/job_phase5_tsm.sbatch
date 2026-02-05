#!/bin/bash
#SBATCH --job-name=ffn-phase5-tsm
#SBATCH --partition=gpu
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/phase5_tsm_%j.out
#SBATCH --error=logs/phase5_tsm_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kamarthi.v@northeastern.edu

# Load modules
module load cuda/12.1.1
module load anaconda3/2024.06

# Activate environment
source activate ffn

# Change to project directory
cd ~/FFN-pytorch-Experiment-Subset

# Run Phase 5: Vanilla TSM Training (should complete in ~4-6 hours)
python train_tsm.py \
    --epochs 50 \
    --batch_size 8 \
    --num_workers 8 \
    --lr 0.01 \
    --checkpoint_dir checkpoints/vanilla_tsm

echo "Phase 5 complete!"
