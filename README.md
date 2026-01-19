# FFN Reproduction - Experimental Subset

A ground-up reproduction of a targeted subset from the Frame Flexible Network (FFN) paper.

## Original Work

**Paper**: Frame Flexible Network (CVPR 2023)
**arXiv**: [2303.14817](https://arxiv.org/abs/2303.14817)
**Official Repository**: [BeSpontaneous/FFN-pytorch](https://github.com/BeSpontaneous/FFN-pytorch)

All credit for the FFN architecture and methodology belongs to the original authors.

## What This Repository Does

The original FFN paper demonstrates a general framework that solves Temporal Frequency Deviation (TFD) across multiple architectures (TSM, TEA, SlowFast, Uniformer) and datasets (SSv1, SSv2, Kinetics400, HMDB51).

**This repository reproduces a minimal subset**:
- **Architecture**: TSM (Temporal Shift Module) with ResNet-50 backbone
- **Dataset**: Something-Something V2
- **Goal**: Demonstrate TFD collapse in vanilla TSM, then show FFN recovers accuracy

## The Problem: Temporal Frequency Deviation

When video models are trained at one frame count and evaluated at another, accuracy collapses:

| Model | Train Frames | Eval Frames | Accuracy |
|-------|--------------|-------------|----------|
| Vanilla TSM | 16 | 16 | ~61% |
| Vanilla TSM | 16 | 4 | ~31% |
| **FFN-TSM** | 16 | 4 | **~56%** |

FFN fixes this by training with multiple frame counts simultaneously and using specialized BatchNorm parameters.

## Repository Structure

```
FFN-pytorch-Experiment-Subset/
├── src/
│   ├── data/       # Data loading pipeline
│   └── models/     # TSM and FFN architectures
├── tests/          # Test suite
├── database/       # Dataset and labels (gitignored)
└── docs/           # Technical documentation
```

## Purpose

This is a learning project for understanding video action recognition and reproducing research results from scratch.

## License

This reproduction is for educational purposes. See the [original repository](https://github.com/BeSpontaneous/FFN-pytorch) for the official implementation and licensing.
