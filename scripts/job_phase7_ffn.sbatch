#!/bin/bash
#SBATCH --job-name=ffn-phase7-ffn
#SBATCH --partition=gpu
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/phase7_ffn_%j.out
#SBATCH --error=logs/phase7_ffn_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kamarthi.v@northeastern.edu

# Use pip --user installed packages
export PATH="$HOME/.local/bin:$PATH"

# Change to project directory
cd ~/ondemand/FFN-pytorch-Experiment-Subset

# Run FFN Training - target 50 epochs
# If checkpoint exists, it resumes automatically
# If job gets killed at 8 hours, checkpoint saves progress
# Just resubmit this same script to continue!

CHECKPOINT_FILE="checkpoints/ffn/latest.pth"

if [ -f "$CHECKPOINT_FILE" ]; then
    echo "Resuming from checkpoint..."
    python3 train_ffn.py \
        --video_dir database/data/20bn-something-something-v2 \
        --labels_dir database/labels \
        --epochs 50 \
        --batch_size 8 \
        --num_workers 8 \
        --lr 0.01 \
        --lambda_kl 1.0 \
        --checkpoint_dir checkpoints/ffn \
        --resume checkpoints/ffn/latest.pth
else
    echo "Starting fresh training..."
    python3 train_ffn.py \
        --video_dir database/data/20bn-something-something-v2 \
        --labels_dir database/labels \
        --epochs 50 \
        --batch_size 8 \
        --num_workers 8 \
        --lr 0.01 \
        --lambda_kl 1.0 \
        --checkpoint_dir checkpoints/ffn
fi

echo "Job complete! Check if all 50 epochs finished. If not, resubmit this script."
