# Phase 7 Pre-Cluster: FFN Training Infrastructure

**Status:** Complete
**Date:** January 2026

---

## Overview

Phase 7 Pre-Cluster builds the training infrastructure for Frame Flexible Network. This includes the FFNTrainer class, training script, and comprehensive testing to ensure everything works before cluster submission.

---

## What Changed from Phase 5 (Vanilla TSM)

| Aspect | Vanilla TSM (Phase 5) | FFN (Phase 7) |
|--------|----------------------|---------------|
| Data Loading | Single frame count | Three frame counts (4F, 8F, 16F) |
| Forward Pass | One input, one output | Three inputs, three outputs |
| Loss Function | Cross-entropy only | CE + Temporal Distillation |
| Validation | Single accuracy | Per-frame-count accuracies |
| Model State | One set of BN stats | Three sets of BN stats |

---

## Component 1: FFNTrainer Class

**File:** `src/training/ffn_trainer.py`

### Why Separate from Vanilla Trainer

The FFN training loop is fundamentally different:
1. **Multi-frame forward pass**: Processes three inputs simultaneously
2. **Complex loss computation**: CE on 16F + KL distillation from 16F to 4F/8F
3. **Per-frame-count metrics**: Tracks accuracy at each frame count separately
4. **TFD evaluation**: Special evaluation mode for measuring frame count gap

Creating a separate class keeps each trainer focused and avoids conditional complexity.

### Key Methods

```python
class FFNTrainer:
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch with multi-frame inputs.

        For each batch:
        1. Load (v_4, v_8, v_16, labels)
        2. Forward: out_4, out_8, out_16 = model(v_4, v_8, v_16, training=True)
        3. Loss: CE(out_16, labels) + λ·KL(out_4||out_16) + λ·KL(out_8||out_16)
        4. Backward and step
        5. Track per-frame-count accuracy
        """

    def validate(self, frame_count: Optional[int] = None) -> Dict[str, float]:
        """
        Run validation at specified frame count(s).

        If frame_count is None, validates at all three (4, 8, 16).
        Each validation uses the matching specialized BatchNorm.
        """

    def train(self, epochs: int, ...) -> None:
        """
        Full training loop with validation and checkpointing.

        - Validates every N epochs at all frame counts
        - Saves best model based on 16F accuracy
        - Tracks best accuracy for each frame count
        """
```

### Training Epoch Flow

```
For each batch (v_4, v_8, v_16, labels):
    ┌─────────────────────────────────────────┐
    │  Forward pass (training=True)           │
    │  ┌─────────────────────────────────┐    │
    │  │ v_4  → [Shared Conv + BN_4]  → out_4 │
    │  │ v_8  → [Shared Conv + BN_8]  → out_8 │
    │  │ v_16 → [Shared Conv + BN_16] → out_16│
    │  └─────────────────────────────────┘    │
    └─────────────────────────────────────────┘
                        ↓
    ┌─────────────────────────────────────────┐
    │  Loss computation                       │
    │  L_CE = CrossEntropy(out_16, labels)    │
    │  L_KL = KL(out_4 || out_16.detach())    │
    │       + KL(out_8 || out_16.detach())    │
    │  L_total = L_CE + λ * L_KL              │
    └─────────────────────────────────────────┘
                        ↓
    ┌─────────────────────────────────────────┐
    │  Backward + Optimizer step              │
    │  - 16F path gets CE gradients           │
    │  - 4F/8F paths get KL gradients         │
    │  - Shared conv gets all gradients       │
    └─────────────────────────────────────────┘
```

### Validation at Different Frame Counts

```python
def _validate_single_frame_count(self, frame_count: int):
    """
    Validate at a specific frame count.

    Uses the model in inference mode:
    - model(x_4=v, training=False) for 4F
    - model(x_8=v, training=False) for 8F
    - model(x_16=v, training=False) for 16F

    Each uses its specialized BatchNorm statistics.
    """
```

---

## Component 2: TFD Evaluation

**Function:** `evaluate_tfd()` in `src/training/ffn_trainer.py`

### What It Measures

Temporal Frequency Deviation (TFD) is the accuracy gap between different frame counts:
- TFD Gap = Acc(16F) - Acc(4F)

For vanilla TSM, this gap is large (e.g., 30 percentage points).
For FFN, this gap should be small (e.g., <5 percentage points).

### Implementation

```python
def evaluate_tfd(model, val_loader, device) -> Dict[str, float]:
    """
    Evaluate accuracy at each frame count.

    Returns:
        {
            'acc_4f': float,   # 4-frame accuracy
            'acc_8f': float,   # 8-frame accuracy
            'acc_16f': float,  # 16-frame accuracy
            'tfd_gap': float,  # 16F - 4F gap
        }
    """
    model.eval()

    for frame_count in [4, 8, 16]:
        # For each frame count, use corresponding input
        # and specialized BatchNorm
        for v_4, v_8, v_16, labels in val_loader:
            if frame_count == 4:
                outputs = model(x_4=v_4.to(device), training=False)
            elif frame_count == 8:
                outputs = model(x_8=v_8.to(device), training=False)
            else:
                outputs = model(x_16=v_16.to(device), training=False)

            # Compute accuracy...
```

---

## Component 3: Training Script

**File:** `train_ffn.py`

### Usage

```bash
# Full training
python train_ffn.py --video_dir database/data/20bn-something-something-v2 \
                    --labels_dir database/labels \
                    --epochs 50 \
                    --batch_size 8

# Test run (2 batches, for validation)
python train_ffn.py --video_dir database/data/20bn-something-something-v2 \
                    --labels_dir database/labels \
                    --test_run

# Evaluate only (no training)
python train_ffn.py --video_dir database/data/20bn-something-something-v2 \
                    --labels_dir database/labels \
                    --eval_only \
                    --resume checkpoints/ffn/best.pth
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--video_dir` | required | Path to SSv2 videos |
| `--labels_dir` | required | Path to label JSON files |
| `--epochs` | 50 | Training epochs |
| `--batch_size` | 8 | Batch size per GPU |
| `--lr` | 0.01 | Learning rate |
| `--lambda_kl` | 1.0 | KL loss weight |
| `--num_frames_h` | 16 | High frame count |
| `--num_frames_m` | 8 | Medium frame count |
| `--num_frames_l` | 4 | Low frame count |
| `--test_run` | False | Run with 2 batches |
| `--eval_only` | False | Only evaluate |
| `--resume` | None | Checkpoint to resume |

### Script Flow

```
1. Parse arguments
2. Select device (CUDA > MPS > CPU)
3. Create dataloaders (SSv2MultiFrameDataset)
4. Create model (TSMFFN with pretrained backbone)
5. Create loss (TemporalDistillationLoss)
6. Create optimizer (SGD) and scheduler (CosineAnnealingLR)
7. Create trainer (FFNTrainer)
8. Resume from checkpoint if specified
9. Train for specified epochs
10. Final TFD evaluation
```

---

## Component 4: Optimizer and Scheduler

### Optimizer: SGD with Momentum

```python
def create_ffn_optimizer(
    model: nn.Module,
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
) -> SGD:
    """
    Standard SGD optimizer per FFN paper.

    All parameters (shared convs, specialized BNs, adaconvs)
    are optimized together.
    """
    return SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
```

### Scheduler: Cosine Annealing

```python
def create_ffn_scheduler(
    optimizer: Optimizer,
    epochs: int = 50,
) -> CosineAnnealingLR:
    """
    Cosine decay over training epochs.

    LR starts at 0.01, decays smoothly to 0 by epoch 50.
    """
    return CosineAnnealingLR(optimizer, T_max=epochs)
```

---

## Component 5: Checkpointing

### What Gets Saved

```python
{
    'epoch': int,                    # Current epoch
    'model_state_dict': dict,        # All model weights
    'optimizer_state_dict': dict,    # Optimizer state
    'scheduler_state_dict': dict,    # LR scheduler state
    'best_acc': float,               # Best 16F accuracy
    'best_acc_4f': float,            # Best 4F accuracy
    'best_acc_8f': float,            # Best 8F accuracy
    'lambda_kl': float,              # KL weight used
}
```

### Checkpoint Files

```
checkpoints/ffn/
├── best.pth          # Best model (highest 16F val accuracy)
├── epoch_5.pth       # Checkpoint at epoch 5
├── epoch_10.pth      # Checkpoint at epoch 10
├── ...
└── final.pth         # Final model after training
```

### Resume Training

```python
# In FFNTrainer.resume():
def resume(self, checkpoint_path: str) -> int:
    """
    Load checkpoint and restore training state.

    Returns epoch to resume from.
    """
    info = load_checkpoint(
        filepath=checkpoint_path,
        model=self.model,
        optimizer=self.optimizer,
        scheduler=self.scheduler,
        device=self.device,
    )
    self.best_acc = info['best_acc']
    self.best_acc_4f = info['extra'].get('best_acc_4f', 0.0)
    self.best_acc_8f = info['extra'].get('best_acc_8f', 0.0)
    return info['epoch'] + 1
```

---

## Data Pipeline Integration

### SSv2MultiFrameDataset

The dataset returns all three frame counts for each video:

```python
# Each __getitem__ returns:
(video_4f, video_8f, video_16f, label)

# Shapes:
# video_4f:  [3, 4, 224, 224]
# video_8f:  [3, 8, 224, 224]
# video_16f: [3, 16, 224, 224]
# label: int
```

### DataLoader Batch Format

```python
# After DataLoader:
for v_4, v_8, v_16, labels in train_loader:
    # v_4:    [B, 3, 4, 224, 224]
    # v_8:    [B, 3, 8, 224, 224]
    # v_16:   [B, 3, 16, 224, 224]
    # labels: [B]
```

---

## Testing Strategy

### Test Coverage

All FFN training components are tested in `tests/test_ffn.py`:

1. **TSM Tests**: Shape preservation, channel shifting
2. **FFNBottleneck Tests**: Training mode, inference mode, specialized BN, adaconv
3. **TSMFFN Tests**: Forward pass, gradient flow
4. **Loss Tests**: CE+KL computation, teacher detachment
5. **Integration Tests**: Full forward/backward, multi-frame inference

### Pre-Cluster Verification

Before submitting to cluster, verify:

```bash
# All tests pass
pytest tests/ -v

# CPU-only mode works
CUDA_VISIBLE_DEVICES="" pytest tests/ -v

# Test run completes
python train_ffn.py --video_dir database/data/20bn-something-something-v2 \
                    --labels_dir database/labels \
                    --test_run
```

---

## File Structure Summary

```
src/training/
├── __init__.py              # Updated with FFN exports
├── utils.py                 # Shared utilities (unchanged)
├── checkpoint.py            # Save/load (unchanged)
├── trainer.py               # Vanilla TSM trainer (unchanged)
└── ffn_trainer.py           # NEW: FFN trainer
    ├── create_ffn_optimizer()
    ├── create_ffn_scheduler()
    ├── FFNTrainer class
    └── evaluate_tfd()

train_ffn.py                 # NEW: FFN training script
```

---

## Expected Training Output

```
Using device: cuda
Loading datasets...
Creating FFN model...

FFN Training on cuda
Training samples: 168913
Validation samples: 24777
Epochs: 50
Lambda KL: 1.0
------------------------------------------------------------

Epoch [1/50] LR: 0.010000
  Batch [100/21114] Loss: 5.1234 (CE: 5.0123, KL: 0.1111) Time: 45.2s
    Acc@1 - 4F: 1.23% | 8F: 1.45% | 16F: 1.67%
  ...
  Train - Loss: 4.8765 (CE: 4.7654, KL: 0.1111)
    Acc@1 - 4F: 2.34% | 8F: 2.56% | 16F: 2.78%
  Validation:
    4F:  Acc@1: 3.45%
    8F:  Acc@1: 3.67%
    16F: Acc@1: 3.89%
  New best 16F accuracy: 3.89%

...

============================================================
Final TFD Evaluation
============================================================
  4F Accuracy:  58.50%
  8F Accuracy:  60.20%
  16F Accuracy: 61.30%
  TFD Gap (16F - 4F): 2.80 points
```

---

## Key Insights

1. **Separate Trainers**: FFN's multi-frame nature justifies a dedicated trainer class
2. **Loss Decomposition**: Tracking CE and KL separately helps debug training issues
3. **Per-Frame Metrics**: Essential for understanding TFD reduction during training
4. **Checkpoint Completeness**: Saving all frame-count accuracies enables proper analysis

---

## Next Steps (Post-Cluster)

Phase 7 Post-Cluster will cover:
1. Submitting training job to SMILE Lab cluster
2. Monitoring training progress
3. Analyzing results and comparing to paper
4. Final TFD evaluation and report

---

*Phase 7 Pre-Cluster Complete - FFN training infrastructure ready for cluster*
