# Phase 5: Vanilla TSM Training - Pre-Cluster Breakdown

## Overview

Phase 5 implements the complete training infrastructure for vanilla TSM on Something-Something V2. This creates the baseline model that will demonstrate TFD (Temporal Frequency Deviation) before FFN fixes it.

---

## What Phase 5 Does

Trains vanilla TSM at a **fixed frame count (16 frames)** to:
1. Train a working video classifier
2. Create a baseline to demonstrate TFD later

---

## Training Pipeline Flow

```
Video File (.webm)
       ↓
[SSv2Dataset] loads frames, applies transforms
       ↓
[DataLoader] batches videos: (B, 3, T, 224, 224)
       ↓
[TSMResNet50] processes through backbone with temporal shifts
       ↓
[Predictions] → (B, 174) logits
       ↓
[CrossEntropyLoss] computes loss against labels
       ↓
[SGD Optimizer] updates weights
       ↓
[CosineAnnealingLR] decays learning rate
       ↓
[Checkpoint] saves model state periodically
```

---

## Files Built

| File | Purpose |
|------|---------|
| `src/training/utils.py` | Device selection (CUDA/MPS/CPU), AverageMeter for tracking metrics, accuracy computation |
| `src/training/checkpoint.py` | Save/load model+optimizer+scheduler state |
| `src/training/trainer.py` | Complete training loop with validation, logging, checkpointing |
| `train_tsm.py` | Main script with CLI arguments for cluster submission |

---

## Key Components Explained

### 1. Device Selection (`get_device()`)

Automatically selects the best available device:
- CUDA (NVIDIA GPUs) - cluster
- MPS (Apple Silicon) - local Mac development
- CPU - fallback

This ensures the same code runs on your Mac and the cluster without modification.

### 2. AverageMeter

Tracks running averages of metrics during training:
```python
meter = AverageMeter("loss")
meter.update(0.5)  # First batch loss
meter.update(0.3)  # Second batch loss
print(meter.avg)   # 0.4 - running average
```

Used to report epoch-level loss and accuracy from batch-level values.

### 3. Accuracy Computation

Computes top-k accuracy:
- **Top-1**: Is the highest-scored class correct?
- **Top-5**: Is the correct class in the top 5 predictions?

SSv2 has 174 classes, so top-5 is a useful secondary metric.

### 4. Checkpointing

Saves complete training state:
- Model weights
- Optimizer state (momentum buffers)
- Scheduler state (current learning rate position)
- Epoch number
- Best accuracy achieved

This allows resuming training if interrupted (common on shared clusters).

### 5. Trainer Class

Orchestrates the complete training loop:
```python
trainer = Trainer(model, train_loader, val_loader, optimizer, scheduler, device)
trainer.train(epochs=50, validate_every=1, save_every=5)
```

Handles:
- Epoch iteration
- Batch processing
- Validation runs
- Checkpoint saving
- Progress logging

---

## Hyperparameters (from paper)

```python
Optimizer: SGD
  - momentum: 0.9
  - weight_decay: 5e-4

Learning Rate: 0.01
  - Schedule: CosineAnnealingLR (decays to ~0 over training)

Batch Size: 8 per GPU
Epochs: 50
Frames: 16 (fixed for vanilla TSM)
```

---

## Pre-Cluster Checklist

Before submitting to cluster:

- [ ] All tests pass: `pytest tests/ -v`
- [ ] CPU mode works: `CUDA_VISIBLE_DEVICES="" pytest tests/`
- [ ] Single batch runs without error
- [ ] Paths are configurable via CLI arguments
- [ ] Checkpoints save and load correctly

---

## Cluster Command

```bash
python train_tsm.py \
    --data_dir /path/to/database \
    --epochs 50 \
    --batch_size 8 \
    --num_workers 4 \
    --checkpoint_dir /path/to/checkpoints
```

---

## Test Files

| Test File | What It Tests |
|-----------|---------------|
| `tests/test_training.py` | Training utilities, optimizer, scheduler, training steps |
| `tests/test_checkpoint.py` | Save/load checkpoint round-trip |
| `tests/test_integration.py` | Full data→model→loss pipeline |

Run all tests before cluster submission:
```bash
pytest tests/test_training.py tests/test_checkpoint.py tests/test_integration.py -v
```

---

## Key Insight

**Why train vanilla TSM first?**

1. **Proves your pipeline works** - if vanilla TSM doesn't train, FFN won't either
2. **Creates TFD baseline** - you need to show the problem before showing FFN solves it
3. **Validates data loading** - confirms videos load correctly at scale
4. **Establishes baseline accuracy** - gives you a number to compare FFN against
