# FFN Reproduction: Step-by-Step Reference

**Goal:** Reproduce Something-Something V2 Results — vanilla TSM collapses from ~61% (16F) to ~31% (4F), FFN recovers to ~56% (4F) and improves 16F to ~64%.

---

## Compute Requirements

| Phase | Where to Run | Notes |
|-------|--------------|-------|
| 1-4 | Local Mac | Setup, data, code implementation |
| **5** | **GPU Cluster** | Train vanilla TSM (50 epochs × 169k videos) |
| 6 | Local Mac | Build FFN components |
| **7** | **GPU Cluster** | Train FFN (50 epochs × 169k videos × 3 frame counts) |
| 8 | Cluster or Mac | Validation/evaluation (faster on cluster) |

**What's an epoch?** One pass through all 168,913 training videos. At batch size 8, that's ~21,114 batches per epoch. 50 epochs = model sees each video 50 times.

---

## Phase 1: Setup

- [X] 1.1. Clone official repo: `git clone https://github.com/BeSpontaneous/FFN-pytorch.git`
- [X] 1.2. Create your own project folder separately — write code from scratch, reference their repo.
- [X] 1.3. Create Python venv with PyTorch (MPS on Mac / CUDA on cluster). Install decord, einops, tensorboard. Verify GPU.

---

## Phase 2: Data

- [X] 2.1. Apply for Something-Something V2 at Qualcomm's AI datasets site. Takes 1-2 days.
- [X] 2.2. Download and extract videos (~20GB) and label CSVs.
- [X] 2.3. Verify: ~169k train, ~25k val, 174 classes.

---

## Phase 3: Data Loading

- [X] 3.1. Implement video loader with decord. Reference: `2D_Network/ops/dataset.py`
- [X] 3.2. Implement uniform frame sampling — divide video into T segments, sample center of each.
- [X] 3.3. Implement augmentations — random crop 224, NO horizontal flip (directional actions), ImageNet normalize. Reference: `2D_Network/ops/transforms.py`
- [X] 3.4. Build single-frame Dataset returning `(3, T, 224, 224)` and label.
- [X] 3.5. Build multi-frame Dataset for FFN — returns v_L (4F), v_M (8F), v_H (16F) from same video. Reference: `2D_Network/ops/dataset_3sequences.py`

---

## Phase 4: Vanilla TSM

- [X] 4.1. Understand TSM: shifts 1/8 channels forward, 1/8 backward, rest unchanged. Zero extra params. Reference: `2D_Network/ops/temporal_shift.py`
- [X] 4.2. Implement ResNet-50 backbone with torchvision pretrained weights.
- [X] 4.3. Insert TSM after first conv in each residual block. Reference: `2D_Network/ops/backbone/resnet_TSM.py`
- [X] 4.4. Add classifier: global avg pool → temporal consensus (avg across T) → FC(2048 → 174). Reference: `2D_Network/ops/models.py`
- [X] 4.5. Verify forward: input `(B, 3, 16, 224, 224)` → output `(B, 174)`. Verify backward works.
- [X] 4.6. Confirm ~25.6M parameters. (Actual: 23.86M - difference from FC layer 1000→174 classes)

---

## Phase 5: Train Vanilla TSM

**CLUSTER REQUIRED** — 50 epochs x 169k videos. Expect 1-2 days on GPU cluster, 1-2 weeks on local Mac.

- [X] 5.1. Hyperparams: SGD, momentum 0.9, weight decay 5e-4, LR 0.01, MultiStepLR decay at epochs 20 and 40 (gamma=0.1), batch 32, 50 epochs, AMP enabled, gradient clipping max_norm=20. GPU: 1x H200 (equivalent to paper's 2 standard GPUs). Reference: `2D_Network/exp/tsm_sthv2/run.sh`
- [ ] 5.2. Train at 16F only. Save checkpoints. Reference: `2D_Network/main.py`
- [ ] 5.3. Evaluate same checkpoint at 16F (~61%), 8F (~52%), 4F (~31%). This is TFD.

---

## Phase 6: FFN Components

- [X] 6.1. Weight Sharing — single backbone, shared conv weights across all frame counts.
- [X] 6.2. Specialized BatchNorm — private BN per frame count (bn1_4, bn1_8, bn1_16). Convs shared, BN private. Reference: `2D_Network/ops/backbone/resnet_TSM_FFN.py` lines 74-81
- [X] 6.3. Temporal Distillation — KL divergence aligns p_L and p_M toward p_H. CE loss on p_H only. Reference: `2D_Network/main_FFN.py` lines 355-368
- [X] 6.4. Combined loss: `L = CE(p_H, y) + λ·KL(p_L || p_H) + λ·KL(p_M || p_H)`, λ=1.
- [X] 6.5. Weight Alteration — depthwise 1×1 conv adapters (adaconv_4/8/16) with residual connection, near-zero init. Adds ~0.1M params. Reference: `resnet_TSM_FFN.py` lines 90-95

---

## Phase 7: Train FFN

**CLUSTER REQUIRED** — 50 epochs x 169k videos x 3 frame counts per batch. Expect 1-2 days on GPU cluster.

- [X] 7.1. Same hyperparams as vanilla TSM.
- [X] 7.2. Each batch: load v_L, v_M, v_H from same video, forward all three, compute combined loss. Reference: `2D_Network/main_FFN.py` lines 342-368
- [ ] 7.3. Train 50 epochs. Reference: `2D_Network/opts.py` for config, `main_FFN.py` for loop.

---

## Phase 8: Validate FFN

- [ ] 8.1. Evaluate at 16F → expect ~63.61%
- [ ] 8.2. Evaluate at 8F → expect ~61.86%
- [ ] 8.3. Evaluate at 4F → expect ~56.07%
- [ ] 8.4. Compare to vanilla TSM. Build your Table 1. Reference: `main_FFN.py` lines 422-433 for inference sub-network selection.

---

## Key File Map

| Purpose | File |
|---------|------|
| Training loop | `main_FFN.py` |
| Model wrapper | `ops/models_FFN.py` |
| FFN backbone | `ops/backbone/resnet_TSM_FFN.py` |
| Multi-frame dataloader | `ops/dataset_3sequences.py` |
| Hyperparams | `opts.py` |

---

## Debugging Signals

| Symptom | Check |
|---------|-------|
| TFD doesn't appear (4F ≈ 16F) | Not actually using fewer frames, or BN in wrong mode |
| FFN doesn't help | KL loss not decreasing, specialized BN routing broken |
| Loss explodes | LR too high |
| Low accuracy | Wrong labels, augmentation bug, forgot `model.train()` |

---

## Key Equations

- **CE:** cross-entropy on p_H only
- **KL:** `KL(p_L || p_H) + KL(p_M || p_H)`, teacher detached
- **Total:** `L_CE + 1.0 × L_KL`

---

## Key Shapes

| Stage | Shape |
|-------|-------|
| Input | `(B, 3, T, 224, 224)`, T ∈ {4, 8, 16} |
| After backbone | `(B×T, 2048, 7, 7)` |
| After pooling | `(B, 2048)` |
| Output | `(B, 174)` |