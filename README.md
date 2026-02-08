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

| Model | Train Frames | Eval @ 16F | Eval @ 8F | Eval @ 4F |
|-------|:------------:|:----------:|:---------:|:---------:|
| Vanilla TSM | 16 | ~61% | ~TBD% | ~31% |
| **FFN-TSM** | 4, 8, 16 | ~TBD% | ~TBD% | **~56%** |

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

See [FINAL_REPORT.md](FINAL_REPORT.md) for the full experiment report including training configuration, results, and analysis.

## Purpose

Independent research reproduction for the SMILE Lab at Northeastern University. Built from scratch to demonstrate deep understanding of video action recognition and the TFD problem.

## License

Educational use only. See the [official repository](https://github.com/BeSpontaneous/FFN-pytorch) for original licensing.
