# Phase 6: FFN Components Implementation

**Status:** Complete
**Date:** January 2026

---

## Overview

Phase 6 implements the core Frame Flexible Network (FFN) components that solve Temporal Frequency Deviation (TFD). These components transform the vanilla TSM into a frame-flexible model that maintains accuracy across different frame counts.

---

## The TFD Problem (Recap)

When a video model is trained at 16 frames and evaluated at 4 frames:
- BatchNorm statistics computed on 16-frame inputs don't match 4-frame distributions
- Accuracy drops significantly (e.g., 61% -> 31% on SSv2)

FFN solves this with three key innovations.

---

## Component 1: Specialized BatchNorm

**File:** `src/models/tsm_ffn.py`

### What It Does
Instead of one BatchNorm layer per convolution, FFN uses THREE separate BatchNorm layers - one for each frame count (4F, 8F, 16F).

### Implementation Pattern

```python
# In FFNBottleneck.__init__:
# Shared convolutions (same for all frame counts)
self.conv1 = conv1x1(inplanes, planes)
self.conv2 = conv3x3(planes, planes, stride)
self.conv3 = conv1x1(planes, planes * 4)

# Specialized BatchNorm (private per frame count)
self.bn1_4 = nn.BatchNorm2d(planes)   # For 4-frame
self.bn1_8 = nn.BatchNorm2d(planes)   # For 8-frame
self.bn1_16 = nn.BatchNorm2d(planes)  # For 16-frame
# ... same pattern for bn2 and bn3
```

### Why It Works
- Convolutions learn frame-count-agnostic visual features (shared)
- BatchNorm learns frame-count-specific statistics (private)
- Each BN tracks mean/variance appropriate for its frame count
- At inference, use the BN matching the input frame count

### Forward Pass Logic

```python
def forward(self, x_4, x_8, x_16, training=True):
    if training:
        # Process all three in parallel, each with own BN
        out_4 = self.bn1_4(self.conv1(x_4))
        out_8 = self.bn1_8(self.conv1(x_8))
        out_16 = self.bn1_16(self.conv1(x_16))
        ...
    else:
        # Use only the BN matching the input
        if x_4 is not None:
            out = self.bn1_4(self.conv1(x_4))
        elif x_8 is not None:
            out = self.bn1_8(self.conv1(x_8))
        ...
```

---

## Component 2: Weight Alteration (adaconv)

**File:** `src/models/tsm_ffn.py`

### What It Does
Depthwise 1x1 convolutions that adapt shared weights for each frame count, applied with a residual connection.

### Implementation Pattern

```python
# In FFNBottleneck.__init__:
# Depthwise (groups=inplanes means each channel processed independently)
self.adaconv_4 = nn.Conv2d(inplanes, inplanes,
                            kernel_size=1, groups=inplanes, bias=False)
self.adaconv_8 = nn.Conv2d(inplanes, inplanes,
                            kernel_size=1, groups=inplanes, bias=False)
self.adaconv_16 = nn.Conv2d(inplanes, inplanes,
                             kernel_size=1, groups=inplanes, bias=False)

# Near-zero initialization (sigma = 1e-3)
self.adaconv_4.weight.data.normal_(0, 1e-3)
self.adaconv_8.weight.data.normal_(0, 1e-3)
self.adaconv_16.weight.data.normal_(0, 1e-3)
```

### Why It Works
1. **Depthwise = Lightweight**: With `groups=inplanes`, we have C parameters instead of C^2
2. **Near-zero init**: Initial output is almost identity (y = x + ~0 = x)
3. **Residual connection**: `y = x + adaconv(x)` preserves pretrained behavior at start
4. **Learns per-frame adaptation**: During training, each adapter learns frame-specific adjustments

### Application in Forward Pass

```python
# After TSM, before convolutions:
out_4 = self.tsm(x_4)
iden_4 = out_4                    # Save for residual
out_4 = self.adaconv_4(out_4)     # Apply adapter
out_4 = out_4 + iden_4            # Residual connection (Eq. 8)
out_4 = self.conv1(out_4)         # Then shared conv
```

---

## Component 3: Temporal Distillation Loss

**File:** `src/models/temporal_distillation.py`

### What It Does
Knowledge distillation from the 16-frame (teacher) to 4-frame and 8-frame (students).

### Mathematical Formulation

From the FFN paper:

**Equation 4 - KL Divergence:**
```
L_KL = KL(p_L || p_H) + KL(p_M || p_H)
```

**Equation 5 - Total Loss:**
```
L = L_CE(p_H, y) + λ * L_KL
```

Where:
- `p_H` = softmax predictions from 16-frame
- `p_M` = softmax predictions from 8-frame
- `p_L` = softmax predictions from 4-frame
- `y` = ground truth labels
- `λ = 1.0` (from paper)

### Implementation Pattern

```python
class TemporalDistillationLoss(nn.Module):
    def forward(self, output_l, output_m, output_h, target):
        # Cross-entropy on high-frame ONLY
        loss_ce = F.cross_entropy(output_h, target)

        # Teacher probabilities - CRITICAL: detach!
        teacher_probs = F.softmax(output_h.detach(), dim=1)

        # KL divergence for students
        loss_kl_l = F.kl_div(
            F.log_softmax(output_l, dim=1),
            teacher_probs,
            reduction='batchmean'
        )
        loss_kl_m = F.kl_div(
            F.log_softmax(output_m, dim=1),
            teacher_probs,
            reduction='batchmean'
        )

        # Total loss
        return loss_ce + lambda_kl * (loss_kl_l + loss_kl_m)
```

### Critical: The detach() Operation

**Why `output_h.detach()` is essential:**
- Without detach: gradients from KL loss flow backward through the teacher (16F path)
- This creates conflicting signals: CE wants correct class, KL wants student similarity
- With detach: 16F path gets gradients ONLY from CE loss
- Students (4F, 8F) get gradients from KL to match teacher

---

## File Structure Created

```
src/models/
├── __init__.py            # Updated with FFN exports
├── temporal_shift.py      # (Phase 4 - unchanged)
├── tsm.py                 # (Phase 4 - unchanged)
├── tsm_ffn.py             # NEW: FFN model components
│   ├── TSM class          # Temporal Shift for FFN
│   ├── FFNBottleneck      # Block with Specialized BN + adaconv
│   ├── FFNResNet          # Full backbone
│   ├── TSMFFN             # High-level wrapper
│   ├── resnet50_ffn()     # Factory with pretrained loading
│   └── create_ffn_model() # Device-aware factory
└── temporal_distillation.py  # NEW: FFN loss functions
    ├── TemporalDistillationLoss  # Class-based loss
    ├── FFNLoss                   # Alternative interface
    └── compute_ffn_loss()        # Functional interface

tests/
└── test_ffn.py            # NEW: Comprehensive FFN tests
```

---

## Test Coverage

**File:** `tests/test_ffn.py`

### Test Categories

1. **TSM Tests**
   - Shape preservation
   - Different segment counts
   - Channel shifting verification

2. **FFNBottleneck Tests**
   - Training mode (3 inputs -> 3 outputs)
   - Inference mode (1 input -> 1 output)
   - Specialized BN verification
   - Weight Alteration verification
   - Near-zero initialization check
   - Shared convolution verification

3. **FFNResNet Tests**
   - Training output shapes
   - Inference output shapes

4. **TSMFFN Wrapper Tests**
   - Training forward pass
   - Inference at different frame counts
   - Gradient flow verification

5. **Temporal Distillation Loss Tests**
   - Loss computation
   - Gradient flow to students
   - Teacher detachment verification
   - Functional and class-based interfaces

6. **Integration Tests**
   - Full forward/backward pass
   - Train/eval mode switching
   - Multi-frame inference

7. **Edge Case Tests**
   - Missing input handling
   - Batch size 1

---

## Pretrained Weight Loading

When loading ImageNet pretrained weights into FFN:

```python
# Mapping pattern in resnet50_ffn():
# Vanilla: layer1.0.bn1.weight
# FFN:     layer1.0.bn1_4.weight, layer1.0.bn1_8.weight, layer1.0.bn1_16.weight

for k in new_state_dict.keys():
    if k.replace('_4', '') in pretrained_dict:
        new_state_dict[k] = pretrained_dict[k.replace('_4', '')]
    elif k.replace('_8', '') in pretrained_dict:
        new_state_dict[k] = pretrained_dict[k.replace('_8', '')]
    elif k.replace('_16', '') in pretrained_dict:
        new_state_dict[k] = pretrained_dict[k.replace('_16', '')]
```

This copies the same pretrained BN weights to all three specialized BNs, allowing them to diverge during FFN training.

---

## Usage Examples

### Training Mode

```python
from src.models import TSMFFN, TemporalDistillationLoss

model = TSMFFN(num_classes=174, pretrained=True)
criterion = TemporalDistillationLoss(lambda_kl=1.0)

# Forward pass with all three frame counts
out_4, out_8, out_16 = model(v_4, v_8, v_16, training=True)

# Compute loss
loss, loss_dict = criterion(out_4, out_8, out_16, labels)
loss.backward()
```

### Inference Mode

```python
model.eval()
with torch.no_grad():
    # Evaluate at 4 frames
    out = model(x_4=video_4f, training=False)

    # Evaluate at 8 frames
    out = model(x_8=video_8f, training=False)

    # Evaluate at 16 frames
    out = model(x_16=video_16f, training=False)
```

---

## Key Insights

1. **Shared vs Private**: FFN's power comes from sharing what should be shared (convolutions) and privatizing what should be private (BatchNorm statistics)

2. **Initialization Matters**: Near-zero init for adaconv ensures pretrained features aren't destroyed at training start

3. **Gradient Control**: The `detach()` in distillation loss is critical - it prevents the teacher from being pulled toward student predictions

4. **Training Efficiency**: Three sub-networks share most parameters, so FFN training cost is only ~1.5x vanilla (not 3x)

---

## Next Steps

Phase 7 will build the FFN training infrastructure:
- FFNTrainer class (separate from vanilla Trainer)
- train_ffn.py script with multi-frame data loading
- Comprehensive pre-cluster testing

---

*Phase 6 Complete - FFN components ready for training infrastructure*
