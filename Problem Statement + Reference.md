# FFN (Frame Flexible Network) Paper: Technical Reference Guide

## The Core Problem

Video recognition models need different frame counts (4F, 8F, 16F) to serve different compute budgets at inference time.

## Current Practice: ST (Separated Training)

Train 3 separate models — one for each frame count. This gives good accuracy but costs:

1. **3x training runs** (repetitive, time-consuming)
2. **3x storage** (25.6M × 3 parameters instead of ~25.7M)
3. **No runtime flexibility** (can't dynamically switch based on device constraints)

## The Naive Fix (Why It Fails)

Train once at 16F, test at whatever frame count you need.

**Result**: Performance drops significantly at non-trained frame counts.

This failure mode is called **TFD (Temporal Frequency Deviation)**.

## TFD (Temporal Frequency Deviation) Mechanistic Breakdown (Why It Happens)

**Key notation:**
- v^H, v^L = same video sampled at High (16F) vs Low (4F) frame counts (inputs, not models)
- μ, σ = mean and variance (normalization statistics)
- γ, β = learned scale and shift parameters
- x = features, y = normalized output

**Step-by-step TFD scenario:**

1. **Training**: Feed v^H (16F videos) → BN computes and stores μ^H, σ^H, learns γ^H, β^H
2. **Training ends**: These statistics are frozen into the model
3. **Inference**: You feed v^L (4F video) to get a prediction
4. **Problem**: Features x^L go through BN → BN applies frozen μ^H, σ^H → output y^L' is miscalibrated

**The equation (Eq. 1):**
```
y^L' = γ^H * (x^L - μ^H) / sqrt(σ^H² + ε) + β^H
```

**Why it breaks**: x^L (4F features) has different natural statistics than what BN (Batch Normalization) expects (16F statistics). The normalization is calibrated for one frame count but applied to another. This miscalibration propagates through the network → wrong predictions → TFD.

**Figure 3 proof**: Empirically shows μ and σ are measurably different between 4F and 16F at every layer.

**Core Insight**: BN/LN (Layer Normalization) statistics are data-dependent. Different frame counts = different data distributions = different statistics. This is intrinsic to the data, not a model flaw. TFD is unavoidable unless you explicitly account for it.

## What FFN (Frame Flexible Network) Promises

- One training run
- One stored model (~1/3 the parameters of ST)
- Works well at ALL frame counts (even beats ST)
- Runtime flexibility to choose frames based on compute budget

## The Research Question

Why does TFD happen, and how do we fix it?

**Answer preview**: 
- TFD happens because of normalization statistics shift across frame counts
- FFN fixes it with MFAL (Multi-Frequency Alignment) + MFAD (Multi-Frequency Adaptation)

---

## FFN Solution: MFAL (Multi-Frequency Alignment)

**Training setup:** One video → sample at 4F (v^L), 8F (v^M), 16F (v^H) → feed all three through shared weights simultaneously

**Two components:**

### 1. WS (Weight Sharing)
- Same convolution/classifier weights across all three sub-networks (F^L, F^M, F^H)
- Forces model to find parameters θ that work for ALL frame counts
- Inductive bias: same video at different frame rates = same class

### 2. TD (Temporal Distillation)
- **CE (Cross-Entropy) loss on p^H only**: Learn to classify correctly using 16F (best info)
- **KL (Kullback-Leibler) loss on p^L, p^M**: Make them match p^H (teacher-student)
- Avoids conflicting gradients from applying CE to all predictions

**What each loss does:**
- **CE loss (inter-class)**: Separate different action classes from each other
- **KL loss (intra-instance)**: Same video at different frame rates → same prediction

**Total loss:** L = L_CE + λ · L_KL (λ = 1, not tuned)

**Result:** Temporal frequency invariant representations — frame rate doesn't change classification.

---

## FFN Solution: MFAD (Multi-Frequency Adaptation)

**The remaining problem:** Weight sharing enforces invariance, but BN/LN statistics still differ per frame count (the Section 3 diagnosis). Also, one set of weights might not be optimal for ALL frame counts.

**Two components:**

### 1. SN (Specialized Normalization)
- Each frame count gets its own μ*, σ*, γ*, β*
- Fixes the TFD root cause: no more mismatched statistics
- Negligible cost (<1% of model parameters)

### 2. WA (Weight Alteration)
- Small DW (Depth-Wise) conv layer (φ) per frame count
- Transforms shared weights W → specialized W* = φ ⊗ W
- Residual connection: preserves pre-trained behavior if adapter does nothing
- Think of it as a lightweight "decorator" on shared weights

**Figure 5 key insight:**
- **Shared (purple arrows):** Conv, Attention, Feed Forward — heavy compute
- **Specialized (no arrows):** BN/LN, Weight Alteration — lightweight

**Design principle:** Specialize where it's cheap, share where it's expensive.

**Result:** Each frame count has properly calibrated normalization + slight weight customization, while still benefiting from shared representations.