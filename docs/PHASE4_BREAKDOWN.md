# Phase 4: TSM Model Architecture Breakdown

## High-Level Summary

Built a video classification model that combines:
- **ResNet-50**: Pretrained image classifier (ImageNet)
- **TSM (Temporal Shift Module)**: Enables temporal reasoning across frames
- **Temporal Consensus**: Averages predictions across frames

Input: `(B, 3, T, 224, 224)` video tensor
Output: `(B, 174)` class predictions for SSv2

---

## Medium-Level: Architecture Flow

```
Video Input (B, 3, T, 224, 224)
         |
         v
    Reshape to (B*T, 3, 224, 224)   # Treat frames as batch
         |
         v
    ┌────────────────────────────────────┐
    │         ResNet-50 Backbone          │
    │                                     │
    │  conv1 → bn1 → relu → maxpool       │
    │         |                           │
    │         v                           │
    │  layer1: [TSM → conv1 → bn → conv2] │ x3 blocks
    │  layer2: [TSM → conv1 → bn → conv2] │ x4 blocks
    │  layer3: [TSM → conv1 → bn → conv2] │ x6 blocks
    │  layer4: [TSM → conv1 → bn → conv2] │ x3 blocks
    │         |                           │
    │         v                           │
    │  Global Average Pool → (B*T, 2048)  │
    └────────────────────────────────────┘
         |
         v
    Dropout (0.5)
         |
         v
    FC Layer (2048 → 174)
         |
         v
    Reshape to (B, T, 174)
         |
         v
    Temporal Consensus (mean over T)
         |
         v
    Output (B, 174)
```

---

## Low-Level: TSM Operation

### The Core Insight

TSM allows frames to exchange information with zero extra parameters by shifting channels:

```python
def temporal_shift(x, num_frames, shift_div=8):
    # x: (B*T, C, H, W)
    # Reshape to (B, T, C, H, W)
    x = x.view(b, t, c, h, w)

    fold = c // shift_div  # 1/8 of channels

    out = torch.zeros_like(x)

    # Forward shift: frame t sees frame t+1's channels
    out[:, :-1, :fold] = x[:, 1:, :fold]

    # Backward shift: frame t sees frame t-1's channels
    out[:, 1:, fold:2*fold] = x[:, :-1, fold:2*fold]

    # Rest unchanged
    out[:, :, 2*fold:] = x[:, :, 2*fold:]

    return out.view(bt, c, h, w)
```

### Visual Example (4 frames, 8 channels)

```
Before shift:              After shift:
Frame: 0  1  2  3          Frame: 0  1  2  3
  Ch0: A  B  C  D            Ch0: B  C  D  0  ← forward (sees future)
  Ch1: E  F  G  H            Ch1: 0  E  F  G  ← backward (sees past)
Ch2-7: [unchanged]         Ch2-7: [unchanged]
```

Frame 1 now has:
- Channel 0: Frame 2's value (future context)
- Channel 1: Frame 0's value (past context)
- Channels 2-7: Its own values

This mixing happens BEFORE the conv layer, so convolutions now operate on temporally-mixed features.

---

## Key Implementation Details

### TSM Placement

TSM wraps `conv1` of each residual block:

```python
def _insert_tsm(self):
    for block in self.backbone.layer1.children():
        block.conv1 = TemporalShift(block.conv1, num_frames=16)
    # Same for layer2, layer3, layer4
```

### Why After conv1?

The TSM paper tested multiple placements. Inserting after the first conv of each block:
- Adds temporal reasoning early in each block
- Doesn't break the residual connection
- Works with pretrained weights (conv weights unchanged)

### Temporal Consensus

After ResNet outputs per-frame features, we average predictions:

```python
out = out.view(b, t, num_classes)  # (B, T, 174)
out = out.mean(dim=1)               # (B, 174)
```

This is simpler than other fusion methods but works well.

---

## Files Created

| File | Purpose |
|------|---------|
| `src/models/__init__.py` | Package exports |
| `src/models/temporal_shift.py` | TSM function and module |
| `src/models/tsm.py` | TSMResNet50 model class |
| `tests/test_model.py` | Model architecture tests |

---

## Test Results

| Test | Result |
|------|--------|
| temporal_shift function | PASSED |
| TemporalShift module | PASSED |
| Forward pass (B=2, T=16) | PASSED |
| Backward pass (gradients) | PASSED |
| Multi-frame (T=4,8,16) | PASSED |
| Parameter count | 23.86M (expected ~25M) |

The parameter count is slightly lower than vanilla ResNet-50 (~25.6M) because we reduced the FC layer from 1000 classes (ImageNet) to 174 classes (SSv2).

---

## Why This Works

1. **Pretrained features**: ResNet-50 already knows how to extract visual features (edges, textures, objects)

2. **TSM adds temporal awareness**: Channel shifting lets each frame see neighboring frames without any new parameters

3. **Consensus aggregates**: Averaging across frames combines per-frame predictions into a single video-level prediction

4. **TFD vulnerability**: The BatchNorm layers compute statistics assuming a specific frame count (e.g., 16F). When tested at different frame counts (4F), statistics mismatch and accuracy drops. This is the TFD problem FFN will solve.

---

## Next: Phase 5 (Training)

Phase 5 requires GPU cluster to train vanilla TSM on SSv2:
- 50 epochs
- 168,913 training videos
- Batch size 8
- SGD with cosine LR decay

After training, we'll evaluate at 16F, 8F, and 4F to observe TFD collapse.
