#!/bin/bash
# =============================================================================
# FFN Full Experiment Runner
# =============================================================================
# Runs Phase 5 (Vanilla TSM) and Phase 7 (FFN) back-to-back automatically.
# No intervention needed - start it and go to sleep!
#
# Usage:
#   ./scripts/run_full_experiment.sh
#
# This script will:
#   1. Run Vanilla TSM training (50 epochs) - ~4-6 hours on H200
#   2. Automatically start FFN training (50 epochs) - ~10-15 hours on H200
#   3. Log everything to logs/
#
# Total expected time: ~15-20 hours
# =============================================================================

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_ROOT/logs"
CHECKPOINT_DIR="$PROJECT_ROOT/checkpoints"

# Data paths
DATA_DIR="$PROJECT_ROOT/database"
VIDEO_DIR="$DATA_DIR/data/20bn-something-something-v2"
LABELS_DIR="$DATA_DIR/labels"

# Training hyperparameters
EPOCHS=50
BATCH_SIZE=8
NUM_WORKERS=8
LR=0.01
LAMBDA_KL=1.0

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

log_info() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] [INFO]${NC} $1"
}

log_phase() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] [PHASE]${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] [SUCCESS]${NC} $1"
}

# -----------------------------------------------------------------------------
# Main Experiment
# -----------------------------------------------------------------------------

main() {
    cd "$PROJECT_ROOT"

    # Create directories
    mkdir -p "$LOG_DIR"
    mkdir -p "$CHECKPOINT_DIR/vanilla_tsm"
    mkdir -p "$CHECKPOINT_DIR/ffn"

    # Log start
    START_TIME=$(date +%s)
    log_phase "============================================"
    log_phase "FFN FULL EXPERIMENT STARTING"
    log_phase "============================================"
    log_info "Project root: $PROJECT_ROOT"
    log_info "Start time: $(date)"
    log_info "Epochs: $EPOCHS"
    log_info "Batch size: $BATCH_SIZE"
    log_info "Workers: $NUM_WORKERS"
    log_info "Learning rate: $LR"
    echo ""

    # Check GPU
    log_info "Checking GPU availability..."
    python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
    echo ""

    # =========================================================================
    # PHASE 5: Vanilla TSM Training
    # =========================================================================
    log_phase "============================================"
    log_phase "PHASE 5: VANILLA TSM TRAINING"
    log_phase "============================================"
    PHASE5_START=$(date +%s)

    python train_tsm.py \
        --data_dir "$DATA_DIR" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --num_workers $NUM_WORKERS \
        --lr $LR \
        --checkpoint_dir "$CHECKPOINT_DIR/vanilla_tsm" \
        2>&1 | tee "$LOG_DIR/phase5_vanilla_tsm.log"

    PHASE5_END=$(date +%s)
    PHASE5_DURATION=$((PHASE5_END - PHASE5_START))
    log_success "Phase 5 completed in $((PHASE5_DURATION / 3600))h $((PHASE5_DURATION % 3600 / 60))m"
    echo ""

    # =========================================================================
    # PHASE 7: FFN Training
    # =========================================================================
    log_phase "============================================"
    log_phase "PHASE 7: FFN TRAINING"
    log_phase "============================================"
    PHASE7_START=$(date +%s)

    python train_ffn.py \
        --video_dir "$VIDEO_DIR" \
        --labels_dir "$LABELS_DIR" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --num_workers $NUM_WORKERS \
        --lr $LR \
        --lambda_kl $LAMBDA_KL \
        --checkpoint_dir "$CHECKPOINT_DIR/ffn" \
        2>&1 | tee "$LOG_DIR/phase7_ffn.log"

    PHASE7_END=$(date +%s)
    PHASE7_DURATION=$((PHASE7_END - PHASE7_START))
    log_success "Phase 7 completed in $((PHASE7_DURATION / 3600))h $((PHASE7_DURATION % 3600 / 60))m"
    echo ""

    # =========================================================================
    # Summary
    # =========================================================================
    END_TIME=$(date +%s)
    TOTAL_DURATION=$((END_TIME - START_TIME))

    log_phase "============================================"
    log_phase "EXPERIMENT COMPLETE!"
    log_phase "============================================"
    log_success "Total time: $((TOTAL_DURATION / 3600))h $((TOTAL_DURATION % 3600 / 60))m"
    log_info "Phase 5 (Vanilla TSM): $((PHASE5_DURATION / 3600))h $((PHASE5_DURATION % 3600 / 60))m"
    log_info "Phase 7 (FFN): $((PHASE7_DURATION / 3600))h $((PHASE7_DURATION % 3600 / 60))m"
    log_info ""
    log_info "Checkpoints saved to:"
    log_info "  - $CHECKPOINT_DIR/vanilla_tsm/"
    log_info "  - $CHECKPOINT_DIR/ffn/"
    log_info ""
    log_info "Logs saved to:"
    log_info "  - $LOG_DIR/phase5_vanilla_tsm.log"
    log_info "  - $LOG_DIR/phase7_ffn.log"
    log_info ""
    log_info "End time: $(date)"
    log_phase "============================================"

    # Save summary
    cat > "$LOG_DIR/experiment_summary.txt" << EOF
FFN Experiment Summary
======================
Start: $(date -r $START_TIME)
End: $(date -r $END_TIME)
Total Duration: $((TOTAL_DURATION / 3600))h $((TOTAL_DURATION % 3600 / 60))m

Phase 5 (Vanilla TSM): $((PHASE5_DURATION / 3600))h $((PHASE5_DURATION % 3600 / 60))m
Phase 7 (FFN): $((PHASE7_DURATION / 3600))h $((PHASE7_DURATION % 3600 / 60))m

Settings:
  Epochs: $EPOCHS
  Batch Size: $BATCH_SIZE
  Workers: $NUM_WORKERS
  Learning Rate: $LR
  Lambda KL: $LAMBDA_KL
EOF

    log_info "Summary saved to $LOG_DIR/experiment_summary.txt"
}

# Run it!
main "$@"
