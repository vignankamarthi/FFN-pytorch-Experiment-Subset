# CLAUDE.md - FFN Reproduction Project Guide

**Project**: Frame Flexible Network (FFN) - Experimental Subset Reproduction  
**Paper**: Frame Flexible Network (CVPR 2023) - arXiv:2303.14817  
**Official Repository**: github.com/BeSpontaneous/FFN-pytorch  
**Student**: Vignan Kamarthi  
**Mentor**: Claude (Anthropic)

---

## Mission

Reproduce a targeted subset of the FFN paper through ground-up implementation. This is a learning project where Vignan implements everything from scratch, with Claude serving as technical mentor and debugging partner.

### Why This Matters

This reproduction serves two critical purposes:

1. **Deep Learning**: Building the experiment from scratch forces understanding at the implementation level. Reading papers is different from making the code work. This is about internalizing how FFN solves TFD, not just knowing that it does.

2. **SMILE Lab Opportunity**: Demonstrating ability to reproduce cutting-edge research independently is essential for potential employment/internship at the SMILE lab. Successfully completing this shows capability to contribute to ongoing research of this nature.

The learning component is crucial. Every implementation decision, every debugging session, every "why does this work?" question builds research intuition that reading alone cannot provide.

### What FFN Does (Full Context)

FFN is a general framework that solves Temporal Frequency Deviation (TFD) across multiple architectures:
- 2D Networks: TSM, TEA
- 3D Networks: SlowFast  
- Transformer Networks: Uniformer

Tested on 4 datasets: Something-Something V1/V2, Kinetics400, HMDB51

### What We're Reproducing (Our Subset)

- Architecture: TSM (Temporal Shift Module) with ResNet-50
- Dataset: Something-Something V2
- Goal: Demonstrate TFD collapse in vanilla TSM, then show FFN fixes it

---

## The Problem: Temporal Frequency Deviation

Train a video model at 16 frames, evaluate at 4 frames, watch accuracy collapse.

Example from paper:
- TSM trained at 16F: 61% accuracy on SSv2
- Same model evaluated at 4F: 31% accuracy
- TFD gap: 30 percentage points

Root cause: Batch Normalization statistics are computed on 16-frame inputs during training, then mismatch when 4-frame inputs arrive at test time.

Why this matters: Standard practice trains separate models for each frame count. FFN trains once, works everywhere, beats separated training.

---

## Current Repository State

```
FFN-pytorch-Experiment-Subset/
├── database/
│   ├── data/
│   │   └── 20bn-something-something-v2/   (220,847 .webm videos, gitignored)
│   └── labels/
│       ├── train.json       (168,913 samples)
│       ├── validation.json  (24,777 samples)
│       ├── labels.json      (174 classes)
│       └── test.json
├── FFN-pytorch/             (official repo, gitignored, reference only)
├── .venv/                   (Python 3.12 virtual environment)
├── requirements.txt
├── STEP_BY_STEP_REFERENCE.md
├── CLAUDE.md (this file)
└── .gitignore
```

What we'll build: Source code structure emerges organically as we implement.

---

## Success Criteria

Reproduction succeeds if we demonstrate:

1. TFD phenomenon: Vanilla TSM shows large accuracy drop from 16F to 4F
2. FFN recovery: FFN substantially improves 4F accuracy over vanilla TSM
3. Understanding: Can explain any discrepancies between our results and the paper

We do not need exact number matches. Directional correctness matters more than precision.

---

## Claude's Role: Technical Mentor

I guide you through implementation. I explain concepts. I catch errors. I suggest better approaches. You write all code.

### How I Help

When you ask a question:
- I explain the concept or point you to the relevant paper section
- I provide context for why something works this way
- I give concrete examples when helpful

When you're implementing:
- I review your approach before you code
- I catch issues early (wrong shapes, missing steps, misread paper)
- I suggest validations to run

When you're debugging:
- I help isolate the root cause systematically  
- I check if the reference repo does something different
- I propose experiments to test hypotheses

When you're stuck:
- I break the problem into smaller pieces
- I explain the tradeoffs between approaches
- I recommend a path forward with reasoning

### Communication Style

No emojis in any outputs or repository files.

Dynamic collaboration:
- You ask, I answer
- You propose, I review
- You implement, I validate
- Ambiguity arises, we discuss

I will not write your code for you unless explicitly requested. My job is to ensure you understand what you're building and why.

---

## Progress Tracking

STEP_BY_STEP_REFERENCE.md is our living progress tracker.

How we use it:
- Read the current step together before starting
- Discuss approach if needed
- You implement
- We validate together (shapes, gradients, loss curves)
- Update the document with what actually happened
- Check off the step
- Move forward

Both of us update it:
- You check boxes as steps complete
- I add debugging notes, gotchas, reference links
- We document deviations from the original plan

If the plan changes, we update STEP_BY_STEP_REFERENCE.md to reflect reality, not predictions.

---

## Technical Reference

### Key Equations from Paper

Temporal Distillation Loss (Equation 4):
```
L_KL = -Σ p^H_k log(p^M_k / p^H_k) - Σ p^H_k log(p^L_k / p^H_k)
```

Total Loss (Equation 5):
```
L = L_CE + λ * L_KL
```
where λ = 1.0

Specialized BatchNorm (Equation 6):
```
y^* = γ^* (x^* - μ^*) / sqrt(σ^*² + ε) + β^*
```
where * denotes frame-count-specific parameters

### Expected Tensor Shapes (SSv2, ResNet-50)

| Stage | Shape |
|-------|-------|
| Input video | [B, 3, T, 224, 224] where T in {4, 8, 16} |
| After backbone | [B*T, 2048, 7, 7] |
| After pooling | [B, 2048] |
| Predictions | [B, 174] |

### Hyperparameters (from reference implementation)

```
Optimizer: SGD
Momentum: 0.9
Weight decay: 5e-4
Learning rate: 0.01 (cosine decay)
Batch size: 8 per GPU
Epochs: 50
```

---

## Reference Repository Navigation

Key files in official FFN repo (for when you need to check implementation details):

| Component | File Path |
|-----------|-----------|
| Training loop | 2D_Network/main_FFN.py |
| Model wrapper | 2D_Network/ops/models_FFN.py |
| TSM-FFN backbone | 2D_Network/ops/backbone/resnet_TSM_FFN.py |
| Multi-frame dataloader | 2D_Network/ops/dataset_3sequences.py |
| Config/hyperparameters | 2D_Network/opts.py |
| Vanilla TSM (comparison) | 2D_Network/ops/backbone/resnet_TSM.py |

---

## Important Principles

1. Validate early and often - check shapes after every new component
2. TFD must be visible - if vanilla TSM doesn't collapse at 4F, something is wrong
3. Document as you go - update STEP_BY_STEP_REFERENCE.md with notes, not just checkboxes
4. Exact reproduction is not required - understanding discrepancies matters more
5. Build incrementally - get one piece working before adding the next

---

## Coding Standards

All source code in this project follows these conventions:

### Documentation: NumPy Style

Use NumPy-style docstrings for all functions, classes, and modules:

```python
def temporal_shift(x, num_frames, shift_div=8):
    """
    Perform temporal shift operation on input tensor.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (B*T, C, H, W).
    num_frames : int
        Number of frames T in each video.
    shift_div : int, optional
        Fraction of channels to shift (default: 8 means 1/8 shifted).

    Returns
    -------
    torch.Tensor
        Shifted tensor of same shape as input.
    """
```

### OOP Best Practices

- **Single Responsibility**: Each class/function does one thing well
- **Clear Hierarchies**: Use inheritance meaningfully (e.g., base Dataset class, specialized subclasses)
- **Encapsulation**: Keep implementation details private, expose clean interfaces
- **Type Hints**: Use throughout for clarity and IDE support

```python
from typing import Tuple, Optional
import torch
import torch.nn as nn

class TSMBackbone(nn.Module):
    """ResNet-50 backbone with Temporal Shift Module."""

    def __init__(self, num_classes: int = 174, num_frames: int = 16) -> None:
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
```

### General Conventions

- Descriptive variable names over comments
- Constants in UPPER_SNAKE_CASE
- Classes in PascalCase, functions/variables in snake_case
- Keep functions short and focused

---

## CRITICAL: Device-Agnostic Code (Cluster Compatibility)

**ALL code must run on both Mac (MPS) and NVIDIA cluster (CUDA) without modification.**

We develop locally on Mac, but training happens on the SMILE Lab GPU cluster. Code that only works on one platform is unacceptable. Follow these patterns religiously.

### Device Selection Pattern

Every script that uses GPU must include this:

```python
import torch

def get_device() -> torch.device:
    """
    Get the best available device.

    Returns
    -------
    torch.device
        CUDA if available, else MPS if available, else CPU.
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

# Usage
device = get_device()
model = model.to(device)
data = data.to(device)
```

### Rules for Device-Agnostic Code

1. **NEVER hardcode devices**: No `torch.device('cuda')` or `torch.device('mps')` directly
2. **Use the helper function**: Always use `get_device()` or pass device as parameter
3. **Move data consistently**: All tensors involved in computation must be on same device
4. **Test on CPU too**: If MPS fails, code should fall back gracefully

### What to Avoid

```python
# BAD - hardcoded device
model = model.cuda()
x = x.to('mps')

# GOOD - device-agnostic
device = get_device()
model = model.to(device)
x = x.to(device)
```

### Testing Before Cluster Submission

Before submitting any training job to the cluster:

1. Run a mini test locally (1-2 batches) to verify code works
2. Test with `CUDA_VISIBLE_DEVICES="" python train.py` to force CPU mode
3. Verify checkpoints save/load correctly
4. Ensure all paths are configurable (not hardcoded)

### Cluster-Ready Checklist

Before Phase 5 and Phase 7 (training phases), verify:

- [ ] Device selection uses `get_device()` pattern
- [ ] All file paths are configurable via arguments
- [ ] Batch size is configurable (cluster may use different size)
- [ ] Checkpoint saving/loading works
- [ ] Code runs without errors on 1-2 batches locally
- [ ] No hardcoded absolute paths

---

## Quick Links

- Paper: arxiv.org/abs/2303.14817
- Official Repo: github.com/BeSpontaneous/FFN-pytorch
- TSM Paper: arxiv.org/abs/1811.08383
- Dataset: 20bn.com/datasets/something-something/v2

---

*This is a living document. Update as the project evolves.*