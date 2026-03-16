# FFN Reproduction -- Experimental Subset

A ground-up reproduction of a targeted subset from the [Frame Flexible Network](https://arxiv.org/abs/2303.14817) (CVPR 2023).

All credit for the FFN architecture and methodology belongs to the original authors.
Official implementation: [BeSpontaneous/FFN-pytorch](https://github.com/BeSpontaneous/FFN-pytorch)

## What This Reproduces

The FFN paper solves Temporal Frequency Deviation (TFD) across multiple architectures and datasets. This repository reproduces **one subset**:

- **Architecture**: TSM (Temporal Shift Module) with ResNet-50
- **Dataset**: Something-Something V2 (168,913 train / 24,777 val / 174 classes)
- **Goal**: Demonstrate TFD collapse in vanilla TSM, then show FFN recovers it

## The Problem: Temporal Frequency Deviation

Train a video model at 16 frames, evaluate at fewer frames, watch accuracy collapse:

| Model | Train Frames | Eval @ 16F | Eval @ 8F | Eval @ 4F | TFD Gap |
|-------|:------------:|:----------:|:---------:|:---------:|:-------:|
| Vanilla TSM | 16 | 56.68% | 48.93% | 30.13% | 26.55 pts |
| **FFN-TSM** | 4, 8, 16 | **58.85%** | **56.52%** | **50.86%** | **8.00 pts** |

FFN reduces the TFD gap by **70%** (26.55 to 8.00 points). The 4F recovery is the headline result: vanilla TSM at 4F scores 30.13%, FFN at 4F scores **50.86%** -- a **+20.73 point improvement** from frame-count-specific BatchNorm and temporal distillation alone.

Root cause: BatchNorm statistics computed on 16-frame inputs mismatch when fewer frames arrive at test time. FFN fixes this with specialized BatchNorm (private BN per frame count), temporal distillation (KL divergence aligning low-frame predictions to high-frame), and weight alteration (lightweight depthwise adapters).

## Repository Structure

```
FFN-pytorch-Experiment-Subset/
├── src/
│   ├── data/              # Video loading, transforms, dataset classes
│   ├── models/            # TSM, FFN backbone, temporal distillation loss
│   └── training/          # Trainer, FFNTrainer, checkpointing, utilities
├── tests/                 # 144 tests across 8 test files
├── scripts/               # SLURM batch scripts for cluster training/eval
├── train_tsm.py           # Vanilla TSM training entry point
├── train_ffn.py           # FFN training entry point
├── eval_tfd.py            # Unified TFD evaluation (both models, all frame counts)
├── database/              # SSv2 dataset and labels (gitignored)
├── docs/                  # Phase-by-phase technical breakdowns
├── STEP_BY_STEP_REFERENCE.md
├── FINAL_REPORT.md
└── END_TO_END_TESTS.md
```

## Usage

Training and evaluation run on Northeastern's GPU cluster (1x NVIDIA H200). All hyperparameters match the original paper.

```bash
# Phase 5: Train vanilla TSM at 16 frames
python train_tsm.py --epochs 50 --batch_size 32 --use_amp --max_grad_norm 20 --lr_steps 20 40

# Phase 7: Train FFN at 4, 8, 16 frames jointly
python train_ffn.py --video_dir database/data/20bn-something-something-v2 \
    --labels_dir database/labels --epochs 50 --batch_size 32 \
    --use_amp --max_grad_norm 20 --lr_steps 20 40

# Phase 8: Evaluate both models at all frame counts
python eval_tfd.py --video_dir database/data/20bn-something-something-v2 \
    --labels_dir database/labels \
    --tsm_checkpoint checkpoints/tsm/best.pth \
    --ffn_checkpoint checkpoints/ffn/best.pth
```

## Testing

```bash
pytest tests/ -v          # Full suite (144 tests)
```

## Results

| Metric | Vanilla TSM | FFN | Improvement |
|--------|:-----------:|:---:|:-----------:|
| 4F Acc@1 | 30.13% | 50.86% | +20.73 pts |
| 8F Acc@1 | 48.93% | 56.52% | +7.59 pts |
| 16F Acc@1 | 56.68% | 58.85% | +2.17 pts |
| TFD Gap (16F-4F) | 26.55 pts | 8.00 pts | 70% reduction |

Our absolute accuracies are ~4 points below the paper across both models at 16F, attributable to single-GPU training (1x H200 vs. 2-GPU distributed). The offset is consistent, so relative comparisons remain valid. TFD gap and recovery percentages match or exceed the paper's reported values.

See [FINAL_REPORT.md](FINAL_REPORT.md) for the full experiment report including training configuration, architecture details, and discrepancy analysis.

## Purpose

Independent research reproduction for the SMILE Lab at Northeastern University. Built from scratch to demonstrate deep understanding of video action recognition and the TFD problem.

## License

Educational use only. See the [official repository](https://github.com/BeSpontaneous/FFN-pytorch) for original licensing.
