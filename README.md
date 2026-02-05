# FFN Reproduction - Experimental Subset

A ground-up reproduction of a targeted subset from the Frame Flexible Network (FFN) paper.

**Paper**: Frame Flexible Network (CVPR 2023) -- [arXiv:2303.14817](https://arxiv.org/abs/2303.14817)
**Official Repository**: [BeSpontaneous/FFN-pytorch](https://github.com/BeSpontaneous/FFN-pytorch)

All credit for the FFN architecture and methodology belongs to the original authors.

## What This Repository Does

The original FFN paper solves Temporal Frequency Deviation (TFD) across multiple architectures (TSM, TEA, SlowFast, Uniformer) and datasets (SSv1, SSv2, Kinetics400, HMDB51).

**This repository reproduces one subset**:
- **Architecture**: TSM (Temporal Shift Module) with ResNet-50
- **Dataset**: Something-Something V2 (168,913 train / 24,777 val / 174 classes)
- **Goal**: Demonstrate TFD in vanilla TSM, then show FFN recovers it

## The Problem: Temporal Frequency Deviation

Train a video model at 16 frames, evaluate at 4 frames, watch accuracy collapse:

| Model | Train Frames | Eval Frames | Expected Accuracy |
|-------|--------------|-------------|-------------------|
| Vanilla TSM | 16 | 16 | ~61% |
| Vanilla TSM | 16 | 4 | ~31% |
| **FFN-TSM** | 4, 8, 16 | 4 | **~56%** |

FFN fixes this with three techniques: specialized BatchNorm (private BN per frame count), temporal distillation (KL divergence aligning low-frame predictions to high-frame), and weight alteration (lightweight depthwise adapters).

## Repository Structure

```
FFN-pytorch-Experiment-Subset/
├── src/
│   ├── data/           # Video loading, transforms, dataset classes
│   ├── models/         # TSM, FFN backbone, temporal distillation loss
│   └── training/       # Trainer, FFNTrainer, checkpointing, utilities
├── tests/              # 124 tests (data, models, FFN, integration, training, checkpoint)
├── scripts/            # Slurm batch scripts, experiment runner
├── train_tsm.py        # Vanilla TSM training entry point
├── train_ffn.py        # FFN training entry point
├── database/           # Dataset and labels (gitignored)
└── docs/               # Phase-by-phase technical breakdowns
```

## Training

Training runs on Northeastern's GPU cluster (1x H200). All settings match the original paper exactly. See the [Experiment Report](#experiment-report) below for full configuration details.

```bash
# Vanilla TSM (Phase 5)
python train_tsm.py --epochs 50 --batch_size 32 --use_amp --max_grad_norm 20 --lr_steps 20 40

# FFN (Phase 7)
python train_ffn.py --video_dir ... --labels_dir ... --epochs 50 --batch_size 32 --use_amp --max_grad_norm 20 --lr_steps 20 40
```

## Purpose

This is a learning project for understanding video action recognition and reproducing research results from scratch, built toward potential work with the SMILE Lab at Northeastern University.

## License

This reproduction is for educational purposes. See the [official repository](https://github.com/BeSpontaneous/FFN-pytorch) for the original implementation and licensing.

---

## Experiment Report

See [FINAL_REPORT.md](FINAL_REPORT.md) for the full experiment report including training configuration, results, and analysis.
