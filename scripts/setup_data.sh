#!/bin/bash
# =============================================================================
# SSv2 Dataset Setup Script for FFN Reproduction
# =============================================================================
# Downloads and extracts the Something-Something V2 dataset (videos + labels).
#
# Usage:
#   ./scripts/setup_data.sh [--verify] [--extract-only]
#
# This script guides you through:
#   1. Downloading videos (2 zip files, ~20GB total)
#   2. Downloading labels (1 zip file, ~1MB)
#   3. Extracting everything to the right locations
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_ROOT/database/data"
VIDEO_DIR="$DATA_DIR/20bn-something-something-v2"
LABELS_DIR="$PROJECT_ROOT/database/labels"
DOWNLOAD_DIR="$PROJECT_ROOT/downloads"

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${BLUE}[STEP]${NC} $1"; }

# -----------------------------------------------------------------------------
# Verify Installation
# -----------------------------------------------------------------------------
verify() {
    log_info "Verifying dataset installation..."

    local errors=0

    # Check videos
    if [ -d "$VIDEO_DIR" ]; then
        VIDEO_COUNT=$(find "$VIDEO_DIR" -name "*.webm" 2>/dev/null | wc -l | tr -d ' ')
        if [ "$VIDEO_COUNT" -gt 200000 ]; then
            log_info "Videos: $VIDEO_COUNT files"
        else
            log_warn "Videos: Only $VIDEO_COUNT files (expected ~220,847)"
            errors=$((errors + 1))
        fi
    else
        log_error "Videos: Directory not found ($VIDEO_DIR)"
        errors=$((errors + 1))
    fi

    # Check labels
    for file in train.json validation.json labels.json; do
        if [ -f "$LABELS_DIR/$file" ]; then
            log_info "Labels: $file"
        else
            log_error "Labels: $file NOT FOUND"
            errors=$((errors + 1))
        fi
    done

    if [ $errors -eq 0 ]; then
        echo ""
        log_info "============================================"
        log_info "DATASET VERIFICATION PASSED"
        log_info "============================================"
        return 0
    else
        echo ""
        log_error "============================================"
        log_error "DATASET VERIFICATION FAILED ($errors errors)"
        log_error "============================================"
        return 1
    fi
}

# -----------------------------------------------------------------------------
# Extract Dataset
# -----------------------------------------------------------------------------
extract() {
    log_step "Extracting dataset..."

    cd "$DOWNLOAD_DIR"

    # Extract videos
    if ls 20bn-something-something-v2-*.zip 1> /dev/null 2>&1; then
        log_info "Unzipping video archives..."
        for f in 20bn-something-something-v2-*.zip; do
            log_info "  Unzipping $f..."
            unzip -o "$f"
        done

        log_info "Concatenating and extracting videos (this takes ~10-20 min)..."
        mkdir -p "$VIDEO_DIR"
        cat 20bn-something-something-v2-?? | tar -xzf - -C "$DATA_DIR"
        log_info "Videos extracted to $VIDEO_DIR"
    else
        log_warn "No video zip files found in $DOWNLOAD_DIR"
    fi

    # Extract labels
    if ls *labels*.zip 1> /dev/null 2>&1 || ls *-labels.zip 1> /dev/null 2>&1; then
        log_info "Extracting labels..."
        mkdir -p "$LABELS_DIR"
        unzip -o "*labels*.zip" -d "$LABELS_DIR" 2>/dev/null || \
        unzip -o "*.zip" -d "$LABELS_DIR" 2>/dev/null || true

        # Move files if they're in a subdirectory
        if [ -d "$LABELS_DIR/labels" ]; then
            mv "$LABELS_DIR/labels/"* "$LABELS_DIR/"
            rmdir "$LABELS_DIR/labels"
        fi
        log_info "Labels extracted to $LABELS_DIR"
    else
        log_warn "No labels zip found in $DOWNLOAD_DIR"
    fi
}

# -----------------------------------------------------------------------------
# Show Download Instructions
# -----------------------------------------------------------------------------
show_instructions() {
    echo ""
    log_info "============================================"
    log_info "DOWNLOAD INSTRUCTIONS"
    log_info "============================================"
    echo ""
    log_info "1. Go to: https://www.qualcomm.com/developer/software/something-something-v-2-dataset/downloads"
    echo ""
    log_info "2. Download these files to: $DOWNLOAD_DIR"
    echo "   - Something-something_zip1  (~10GB)"
    echo "   - Something-something_zip2  (~10GB)"
    echo "   - Something-Something download package labels  (~1MB)"
    echo ""
    log_info "3. After downloading, run:"
    echo "   ./scripts/setup_data.sh --extract-only"
    echo ""
    log_info "4. Verify with:"
    echo "   ./scripts/setup_data.sh --verify"
    echo ""
    log_info "============================================"

    mkdir -p "$DOWNLOAD_DIR"
    log_info "Download directory created: $DOWNLOAD_DIR"
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
main() {
    cd "$PROJECT_ROOT"

    case "${1:-}" in
        --verify)
            verify
            ;;
        --extract-only)
            extract
            verify
            ;;
        --help)
            echo "Usage: $0 [--verify] [--extract-only]"
            echo ""
            echo "Options:"
            echo "  --verify        Check if dataset is properly installed"
            echo "  --extract-only  Extract downloaded files"
            echo "  (no args)       Show download instructions"
            ;;
        *)
            show_instructions
            ;;
    esac
}

main "$@"
