# Phase 7 Post-Cluster: FFN Training Results

**Status:** Pending (requires cluster access)
**Date:** TBD

---

## Overview

This document will be completed after running FFN training on the SMILE Lab GPU cluster. It will contain:
- Actual training logs and metrics
- Comparison with paper results
- TFD analysis
- Lessons learned

---

## Pre-Cluster Checklist

Before submitting to cluster, verify:

- [ ] All unit tests pass: `pytest tests/ -v`
- [ ] All FFN tests pass: `pytest tests/test_ffn.py -v`
- [ ] CPU mode works: `CUDA_VISIBLE_DEVICES="" pytest tests/ -v`
- [ ] Test run completes: `python train_ffn.py --test_run ...`
- [ ] Checkpoint save/load verified
- [ ] Device-agnostic code confirmed

---

## Cluster Submission

### Expected Command

```bash
# On SMILE Lab cluster
python train_ffn.py \
    --video_dir /path/to/ssv2/videos \
    --labels_dir /path/to/ssv2/labels \
    --epochs 50 \
    --batch_size 8 \
    --checkpoint_dir /path/to/checkpoints/ffn
```

### Expected Resources

- GPU: NVIDIA A100 or similar
- Memory: 32GB+ GPU RAM
- Time: ~24-48 hours for 50 epochs on SSv2

---

## Expected Results (from paper)

### TSM + FFN on Something-Something V2

| Frame Count | Expected Accuracy |
|-------------|-------------------|
| 4 frames    | ~58-60%           |
| 8 frames    | ~60-62%           |
| 16 frames   | ~61-63%           |
| TFD Gap     | <5 points         |

### Comparison: Vanilla TSM vs FFN

| Model | 4F Acc | 16F Acc | TFD Gap |
|-------|--------|---------|---------|
| Vanilla TSM | ~31% | ~61% | ~30 pts |
| TSM + FFN | ~58% | ~61% | ~3 pts |

---

## Sections to Complete Post-Training

### 1. Training Metrics

```
TODO: Add actual training curves
- Loss vs epoch (CE and KL components)
- Accuracy vs epoch (4F, 8F, 16F)
- Learning rate schedule
```

### 2. Final TFD Evaluation

```
TODO: Add actual results
  4F Accuracy:  ??%
  8F Accuracy:  ??%
  16F Accuracy: ??%
  TFD Gap: ?? points
```

### 3. Comparison with Paper

```
TODO: Add comparison table
| Metric | Paper | Ours | Difference |
|--------|-------|------|------------|
| ...    | ...   | ...  | ...        |
```

### 4. Analysis

```
TODO: Add analysis
- Did FFN reduce TFD as expected?
- Any unexpected behaviors?
- Training stability observations
```

### 5. Lessons Learned

```
TODO: Document insights
- What worked well
- What required debugging
- Recommendations for future work
```

---

## Files to Save

After training completes, preserve:

1. **Checkpoints**
   - `checkpoints/ffn/best.pth` - Best model
   - `checkpoints/ffn/final.pth` - Final model

2. **Logs**
   - Training stdout/stderr
   - TensorBoard logs (if enabled)

3. **Results**
   - Final TFD evaluation output
   - Validation accuracy at each checkpoint

---

*This document will be updated after cluster training completes*
