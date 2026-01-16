# Phase 3: Data Loading - Technical Breakdown

---

## High Level: What Does It Do?

**One sentence:** Converts raw video files into GPU-ready tensors that the neural network can process.

**The problem:** You have 220,847 `.webm` video files on disk. PyTorch models expect numerical tensors of a specific shape. Something needs to bridge that gap.

**The solution:** A data pipeline that:
1. Reads video files
2. Samples specific frames (not all frames - too slow)
3. Applies augmentations (crop, resize, normalize)
4. Outputs tensors in the exact shape the model expects

**Input:** Path to a video file like `database/data/20bn-something-something-v2/12345.webm`

**Output:** A tensor of shape `(3, 16, 224, 224)` meaning:
- 3 color channels (RGB)
- 16 frames sampled from the video
- 224x224 pixel resolution per frame

---

## Medium Level: How Components Work Together

```
Video File (.webm)
       |
       v
+------------------+
|   video_loader   |  --> Reads video, samples T frames
+------------------+
       |
       v
   numpy array
   (T, H, W, 3)
       |
       v
+------------------+
|   transforms     |  --> Crop, resize, normalize
+------------------+
       |
       v
   torch.Tensor
   (3, T, 224, 224)
       |
       v
+------------------+
|    dataset       |  --> Pairs tensor with label, handles indexing
+------------------+
       |
       v
+------------------+
|   DataLoader     |  --> Batches samples, shuffles, parallel loading
+------------------+
       |
       v
   Batch ready for model
   (B, 3, T, 224, 224)
```

### Component Responsibilities

| Component | File | Input | Output | Purpose |
|-----------|------|-------|--------|---------|
| Video Loader | `video_loader.py` | File path | numpy `(T, H, W, 3)` | Read video, sample frames |
| Transforms | `transforms.py` | numpy `(T, H, W, 3)` | tensor `(3, T, H, W)` | Augment, normalize, reshape |
| Dataset | `dataset.py` | Index | `(tensor, label)` | Organize samples, apply pipeline |
| DataLoader | PyTorch built-in | Dataset | Batches | Parallelize, shuffle, batch |

### Two Dataset Variants

**SSv2Dataset (Vanilla TSM):**
- Returns one video at one frame count
- Used for training vanilla TSM at 16F
- Output: `(video_tensor, label)`

**SSv2MultiFrameDataset (FFN):**
- Returns same video at THREE frame counts (4F, 8F, 16F)
- Used for FFN training with temporal distillation
- Output: `(v_low, v_med, v_high, label)`

---

## Low Level: Implementation Details

### 1. Frame Sampling (`video_loader.py`)

**Problem:** Videos have variable length (30-100+ frames). Model expects fixed T frames.

**Solution:** Uniform segment sampling

```
Video: [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11]  (12 frames)
Sample T=4 frames:

Segment 1: [f0, f1, f2]     -> sample center -> f1
Segment 2: [f3, f4, f5]     -> sample center -> f4
Segment 3: [f6, f7, f8]     -> sample center -> f7
Segment 4: [f9, f10, f11]   -> sample center -> f10

Result: [f1, f4, f7, f10]
```

**Code logic:**
```python
seg_len = total_frames / num_segments  # 12/4 = 3
for i in range(num_segments):
    center = (seg_len * i + seg_len * (i+1)) / 2
    indices.append(int(center))
```

**Edge case:** Video shorter than T frames -> repeat last frame

---

### 2. Video Loading (`video_loader.py`)

**Why PyAV?** Handles VP9/WebM codec (SSv2's format). Decord had pip issues.

**Process:**
```python
container = av.open(video_path)
for frame in container.decode(video=0):
    img = frame.to_ndarray(format="rgb24")  # (H, W, 3)
    all_frames.append(img)
# Then select only the sampled indices
```

**Why decode all frames?** WebM seeking is unreliable. Decoding all then selecting is more robust.

---

### 3. Transforms (`transforms.py`)

**Training transforms:**
```
RandomResizedCrop(224)  -> Random crop + resize to 224x224
NO horizontal flip      -> SSv2 has directional actions ("push left to right")
ImageNet normalize      -> Subtract mean, divide by std
```

**Validation transforms:**
```
Resize(256)            -> Resize shorter side to 256
CenterCrop(224)        -> Deterministic center crop
ImageNet normalize     -> Same normalization
```

**ImageNet normalization values:**
```python
mean = [0.485, 0.456, 0.406]  # Per-channel means
std = [0.229, 0.224, 0.225]   # Per-channel stds
```

**Why ImageNet stats?** ResNet-50 was pretrained on ImageNet. Using same normalization helps transfer learning.

**Shape transformation:**
```
Input:  (T, H, W, C) = (16, 240, 427, 3)   # numpy, HWC format
Output: (C, T, H, W) = (3, 16, 224, 224)   # torch, CHW format
```

---

### 4. Dataset Classes (`dataset.py`)

**Label mapping:**

```python
# labels.json maps class name -> index
{"Holding something": "16", "Pushing something from left to right": "85", ...}

# train.json has video annotations
{"id": "12345", "template": "Holding [something]", ...}

# Remove brackets to match: "Holding [something]" -> "Holding something"
```

**SSv2Dataset.__getitem__(idx):**
```python
def __getitem__(self, idx):
    video_path, label = self.samples[idx]
    frames = load_video_frames(video_path, self.num_frames)  # numpy
    video_tensor = self.transform(frames)                     # torch
    return video_tensor, label
```

**SSv2MultiFrameDataset.__getitem__(idx):**
```python
def __getitem__(self, idx):
    video_path, label = self.samples[idx]
    # Load same video at 3 different frame counts
    frames_4 = load_video_frames(video_path, 4)
    frames_8 = load_video_frames(video_path, 8)
    frames_16 = load_video_frames(video_path, 16)
    # Transform each
    v_low = self.transform(frames_4)
    v_med = self.transform(frames_8)
    v_high = self.transform(frames_16)
    return v_low, v_med, v_high, label
```

---

### 5. DataLoader (PyTorch Built-in)

**What it does:**
- Batches individual samples together
- Shuffles order (training only)
- Loads data in parallel (num_workers)
- Pins memory for faster GPU transfer

**Configuration:**
```python
DataLoader(
    dataset,
    batch_size=8,        # 8 videos per batch
    shuffle=True,        # Random order each epoch
    num_workers=4,       # 4 parallel loading processes
    pin_memory=True,     # Faster CPU->GPU transfer
    drop_last=True,      # Drop incomplete final batch
)
```

**Output shape with batching:**
```
Single sample: (3, 16, 224, 224)
Batch of 8:    (8, 3, 16, 224, 224)  # B=8 prepended
```

---

## Key Design Decisions

| Decision | Reason |
|----------|--------|
| PyAV over decord | Better pip compatibility for Python 3.12 |
| Decode all frames then select | WebM seeking is unreliable |
| No horizontal flip | SSv2 has directional actions |
| ImageNet normalization | ResNet-50 pretrained on ImageNet |
| Uniform segment sampling | Standard approach for temporal models |
| Separate train/val transforms | Augmentation only during training |

---

## Verification Checklist

Run `python test_data_loading.py` to verify:

- [ ] Frame sampling produces correct number of indices
- [ ] Video loading returns correct shape
- [ ] Transforms output (3, T, 224, 224)
- [ ] Dataset returns correct shapes and valid labels
- [ ] DataLoader batches correctly

All tests should pass before proceeding to Phase 4.
