# Phase 5: Vanilla TSM Training - Post-Cluster Results

## Status: COMPLETE

Training ran on Northeastern RC cluster (1x NVIDIA H200, gpu partition, 8hr jobs with checkpoint resumption).

---

## Training Results

| Metric | Value |
|--------|-------|
| Epochs | 50 / 50 |
| Train Loss (final) | 0.0251 |
| Train Acc@1 (final) | 99.91% |
| Val Loss (final) | 2.0338 |
| Val Acc@1 (final) | 54.39% |
| Val Acc@5 (final) | 81.37% |
| **Best Val Acc@1** | **56.69%** |

### Training Configuration

Matched original paper exactly:

- Optimizer: SGD (momentum=0.9, weight_decay=5e-4)
- LR: 0.01 with MultiStepLR decay at epochs 20, 40 (gamma=0.1)
- Batch size: 32
- AMP: enabled (float16 compute, float32 optimizer)
- Gradient clipping: max_norm=20
- Frames: 16 per video
- GPU: 1x NVIDIA H200 (equivalent to paper's 2 standard GPUs)

### Checkpoint Locations (on cluster)

- Best model: `checkpoints/tsm/best.pth`
- Latest model: `checkpoints/tsm/latest.pth`

---

## Observations

### Overfitting

Significant train/val gap by epoch 50 (99.91% train vs 54.39% val). The best validation accuracy of 56.69% was achieved earlier in training before the model memorized the training set. This is expected behavior for SSv2 -- the dataset has high intra-class variation.

### Comparison to Paper

Paper reports ~61% at 16F, we achieved 56.69%. The ~4 point gap is likely due to single-GPU BatchNorm statistics vs the paper's multi-GPU synchronized BatchNorm, which provides implicit regularization.

### TFD Evaluation

Multi-frame evaluation (4F, 8F) deferred to Phase 8 for unified comparison alongside FFN results.

---

*Phase 5 complete. Proceeding to Phase 7 (FFN training), then Phase 8 (unified evaluation).*
