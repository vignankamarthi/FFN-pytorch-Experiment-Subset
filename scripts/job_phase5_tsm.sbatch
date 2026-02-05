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

# Use pip --user installed packages (no conda in sbatch environment)
export PATH="$HOME/.local/bin:$PATH"

# Change to project directory
cd ~/ondemand/FFN-pytorch-Experiment-Subset

# Print environment info for debugging
echo "Python: $(which python3)"
echo "PyTorch CUDA: $(python3 -c 'import torch; print(torch.cuda.is_available())')"
echo "Working directory: $(pwd)"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Run Phase 5: Vanilla TSM Training (should complete in ~4-6 hours)
python3 train_tsm.py \
    --epochs 50 \
    --batch_size 8 \
    --num_workers 8 \
    --lr 0.01 \
    --checkpoint_dir checkpoints/vanilla_tsm

echo "Phase 5 complete!"
