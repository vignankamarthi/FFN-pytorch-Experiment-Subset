#!/bin/bash
#SBATCH --job-name=ffn-phase8-eval
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/phase8_eval_%j.out
#SBATCH --error=logs/phase8_eval_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kamarthi.v@northeastern.edu

# Use pip --user installed packages
export PATH="$HOME/.local/bin:$PATH"

# Change to project directory
cd ~/ondemand/FFN-pytorch-Experiment-Subset

echo "============================================================"
echo "Phase 8: Unified TFD Evaluation"
echo "============================================================"
echo ""

# --- Part 1: Evaluate Vanilla TSM (TFD Collapse) ---
echo "--- Evaluating Vanilla TSM at 4F, 8F, 16F ---"
echo ""

TSM_CHECKPOINT="checkpoints/vanilla_tsm/best.pth"

if [ -f "$TSM_CHECKPOINT" ]; then
    python3 eval_tfd.py \
        --model_type tsm \
        --checkpoint "$TSM_CHECKPOINT" \
        --video_dir database/data/20bn-something-something-v2 \
        --labels_dir database/labels \
        --batch_size 32 \
        --num_workers 8
else
    echo "WARNING: Vanilla TSM checkpoint not found at $TSM_CHECKPOINT"
    echo "Skipping TSM evaluation."
fi

echo ""
echo ""

# --- Part 2: Evaluate FFN (TFD Recovery) ---
echo "--- Evaluating FFN at 4F, 8F, 16F ---"
echo ""

FFN_CHECKPOINT="checkpoints/ffn/best.pth"

if [ -f "$FFN_CHECKPOINT" ]; then
    python3 eval_tfd.py \
        --model_type ffn \
        --checkpoint "$FFN_CHECKPOINT" \
        --video_dir database/data/20bn-something-something-v2 \
        --labels_dir database/labels \
        --batch_size 32 \
        --num_workers 8
else
    echo "WARNING: FFN checkpoint not found at $FFN_CHECKPOINT"
    echo "Skipping FFN evaluation."
fi

echo ""
echo "============================================================"
echo "Phase 8 evaluation complete."
echo "============================================================"
