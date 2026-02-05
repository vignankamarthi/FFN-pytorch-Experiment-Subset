# Experiment Report

Detailed documentation of the FFN reproduction experiment on Something-Something V2.

---

## Architecture

### Vanilla TSM (Baseline)

ResNet-50 backbone with Temporal Shift Module inserted after the first convolution in each residual block. TSM shifts 1/8 of channels forward in time, 1/8 backward, and leaves the rest unchanged -- adding zero extra parameters for temporal modeling.

- **Parameters**: ~23.86M
- **Input**: `(B, 3, T, 224, 224)` where T = 16 frames
- **Output**: `(B, 174)` class predictions
- **Temporal consensus**: Average pooling across T frames after backbone

### FFN-TSM (Frame Flexible Network)

Same ResNet-50 + TSM backbone, with three additions:

1. **Specialized BatchNorm**: Each residual block has private BN parameters for each frame count (4F, 8F, 16F). Convolution weights are shared across all frame counts.

2. **Weight Alteration**: Lightweight depthwise 1x1 convolution adapters (one per frame count) with near-zero initialization and residual connection. Adds ~0.1M parameters.

3. **Temporal Distillation**: KL divergence loss aligns low-frame (4F, 8F) predictions toward the high-frame (16F) teacher. Teacher gradients are detached.

**Loss function**: `L = CE(p_16F, y) + 1.0 * [KL(p_4F || p_16F) + KL(p_8F || p_16F)]`

---

## Training Configuration

All settings sourced from `FFN-pytorch/2D_Network/exp/tsm_sthv2/run.sh`:

| Setting | Paper | Ours |
|---------|-------|------|
| GPUs | 2 standard | 1x H200 (equivalent compute) |
| Batch size | 32 (16/GPU) | 32 |
| Optimizer | SGD | SGD |
| Momentum | 0.9 | 0.9 |
| Weight decay | 5e-4 | 5e-4 |
| Learning rate | 0.01 | 0.01 |
| LR schedule | MultiStepLR [20, 40], gamma=0.1 | MultiStepLR [20, 40], gamma=0.1 |
| Epochs | 50 | 50 |
| Mixed precision | AMP | AMP |
| Gradient clipping | max_norm=20 | max_norm=20 |
| Dropout | 0.5 | 0.5 |
| Shift fraction | 1/8 channels | 1/8 channels |
| Frame counts | 4, 8, 16 | 4, 8, 16 |
| Lambda (KL weight) | 1.0 | 1.0 |

### LR Schedule

Step decay with gamma=0.1 at milestones:

| Epochs | Learning Rate |
|--------|---------------|
| 1-20 | 0.01 |
| 21-40 | 0.001 |
| 41-50 | 0.0001 |

### Hardware

| | Paper | Ours |
|---|-------|------|
| GPU | 2x standard GPU | 1x NVIDIA H200 (80GB) |
| Cluster | -- | Northeastern University RC |
| Partition | -- | gpu (8hr max) |
| Job strategy | -- | Checkpoint-based resumption |

---

## Implementation Notes

### What We Built From Scratch

Every component was implemented from scratch by studying the paper and reference code, not by copying:

- Video loading pipeline with PyAV (decord equivalent)
- Uniform temporal sampling across T segments
- TSM temporal shift operation
- ResNet-50 backbone with TSM insertion
- FFN specialized BatchNorm routing
- Weight alteration adapters with near-zero init
- Temporal distillation loss (KL divergence)
- Multi-frame dataset (synchronized 4F/8F/16F sampling)
- Training infrastructure with AMP, gradient clipping, checkpoint resumption

### Test Coverage

124 tests covering all components:

| Test File | What It Tests | Count |
|-----------|---------------|-------|
| test_data.py | Video loading, transforms, datasets | 6 |
| test_models.py | TSM shift, ResNet-50 backbone | 6 |
| test_ffn.py | FFN bottleneck, specialized BN, loss, integration | 20 |
| test_integration.py | Data-model interface, batch sizes, memory | 15 |
| test_training.py | Optimizer, scheduler, training loop, AMP, grad clip | 35 |
| test_checkpoint.py | Save/load, device mapping, round-trip | 15 |
| test_scripts.py | Script existence, syntax, structure | 12 |

All tests pass on CPU and MPS (Mac). CUDA-specific AMP test skipped on non-CUDA hardware.

---

## Results

### Phase 5: Vanilla TSM (TFD Demonstration)

| Eval Frames | Paper | Ours |
|-------------|-------|------|
| 16F | 61.00% | -- |
| 8F | ~52% | -- |
| 4F | 31.00% | -- |

**TFD Gap (16F - 4F)**: Paper: 30 points | Ours: --

### Phase 7: FFN (TFD Recovery)

| Eval Frames | Paper | Ours |
|-------------|-------|------|
| 16F | 63.61% | -- |
| 8F | 61.86% | -- |
| 4F | 56.07% | -- |

**FFN 4F improvement over vanilla**: Paper: +25 points | Ours: --

*Results will be filled after cluster training completes.*

---

## Analysis

*(To be completed after training)*

### TFD Phenomenon

- Did vanilla TSM show the expected accuracy collapse from 16F to 4F?
- How does the TFD gap magnitude compare to the paper's 30 points?

### FFN Recovery

- Did FFN substantially improve 4F accuracy over vanilla TSM?
- Are 8F and 16F accuracies also improved?

### Discrepancies

- Any significant deviations from paper numbers?
- Possible sources: single GPU vs multi-GPU BN statistics, dataset version differences, augmentation details

---

## Reproduction Checklist

- [ ] TFD demonstrated: vanilla TSM shows >20 point gap between 16F and 4F
- [ ] FFN recovers: 4F accuracy substantially improved over vanilla TSM
- [ ] Results directionally consistent with paper
- [ ] Discrepancies explained
