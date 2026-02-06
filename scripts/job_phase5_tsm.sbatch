#!/bin/bash
#SBATCH --job-name=ffn-phase5-tsm
#SBATCH --partition=gpu
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/phase5_tsm_%j.out
#SBATCH --error=logs/phase5_tsm_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kamarthi.v@northeastern.edu

# Use pip --user installed packages
export PATH="$HOME/.local/bin:$PATH"

# Change to project directory
cd ~/ondemand/FFN-pytorch-Experiment-Subset

# Run Phase 5: Vanilla TSM Training
# Settings match original paper: batch 32, step LR decay, AMP, grad clipping
# Auto-resume from checkpoint if previous job was killed at 8-hour wall

CHECKPOINT_FILE="checkpoints/vanilla_tsm/latest.pth"

if [ -f "$CHECKPOINT_FILE" ]; then
    echo "Resuming from checkpoint..."
    python3 train_tsm.py \
        --epochs 50 \
        --batch_size 32 \
        --num_workers 8 \
        --lr 0.01 \
        --lr_steps 20 40 \
        --use_amp \
        --max_grad_norm 20 \
        --checkpoint_dir checkpoints/vanilla_tsm \
        --resume "$CHECKPOINT_FILE"
else
    echo "Starting fresh training..."
    python3 train_tsm.py \
        --epochs 50 \
        --batch_size 32 \
        --num_workers 8 \
        --lr 0.01 \
        --lr_steps 20 40 \
        --use_amp \
        --max_grad_norm 20 \
        --checkpoint_dir checkpoints/vanilla_tsm
fi

echo "Job complete! Check if all 50 epochs finished. If not, resubmit this script."
