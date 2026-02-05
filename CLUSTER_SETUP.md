# Cluster Setup Guide - FFN Reproduction

**Cluster**: Northeastern University Research Computing (Open OnDemand)
**URL**: https://ood.explorer.northeastern.edu
**User**: kamarthi.v

This guide walks through setting up and running the FFN experiment on the cluster using **checkpoint-based job chaining** to work within partition time limits.

**IMPORTANT**: The OOD VSCode terminal runs in an Apptainer container where conda/modules don't work. Use `pip3 --user` instead.

---

## Table of Contents

1. [Understanding the Strategy](#understanding-the-strategy)
2. [Quick Reference - Partition Settings](#quick-reference---partition-settings)
3. [Phase A: Initial Setup (One-Time)](#phase-a-initial-setup-one-time)
4. [Phase B: Download Dataset](#phase-b-download-dataset)
5. [Phase C: Verify Everything Works](#phase-c-verify-everything-works)
6. [Phase D: Run Training (Job Chaining)](#phase-d-run-training-job-chaining)
7. [Phase E: Get Results Back](#phase-e-get-results-back)
8. [Troubleshooting](#troubleshooting)

---

## Understanding the Strategy

### The Constraint

The `gpu` partition (no approval needed) has an **8-hour maximum** runtime. Our training needs ~15-20 hours total.

### The Solution: Checkpoint-Based Job Chaining

Instead of one long job, we run multiple 8-hour jobs that **resume from checkpoints**:

```
┌─────────────────────────────────────────────────────────────────┐
│  CHECKPOINT-BASED JOB CHAINING STRATEGY                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Phase 5: Vanilla TSM (~4-6 hours)                              │
│  ├── Job 1: Epochs 1-50 → FITS IN ONE 8-HOUR JOB ✓              │
│                                                                 │
│  Phase 7: FFN Training (~10-15 hours)                           │
│  ├── Job 2: Epochs 1-25  → saves checkpoint                     │
│  └── Job 3: Epochs 26-50 → resumes from checkpoint              │
│                                                                 │
│  Total: 2-3 job submissions, each under 8 hours                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Why this works:**
- PyTorch checkpoints save model weights, optimizer state, and current epoch
- Each job picks up exactly where the last one stopped
- No training progress is lost

---

## Quick Reference - Partition Settings

### For Interactive Setup (VSCode)

Use `short` partition (CPU) for setup - more available, no GPU needed for setup:

```
┌─────────────────────────────────────────────────────────────────┐
│  VSCODE SESSION (Setup)                                         │
├─────────────────────────────────────────────────────────────────┤
│  Partition:           short (CPU, 48hr max)                     │
│  Time:                2-4 hours                                 │
│  CPUs:                8                                         │
│  Memory:              32 GB                                     │
│  NOTE: Conda/modules DON'T WORK - use pip3 --user               │
└─────────────────────────────────────────────────────────────────┘
```

### For Training Jobs (Slurm Batch)

Use `gpu` partition for actual training:

```
┌─────────────────────────────────────────────────────────────────┐
│  TRAINING JOBS (Slurm sbatch)                                   │
├─────────────────────────────────────────────────────────────────┤
│  Partition:           gpu                                       │
│  Time:                8 hours (max allowed)                     │
│  GPUs:                1                                         │
│  CPUs:                8                                         │
│  Memory:              32 GB                                     │
└─────────────────────────────────────────────────────────────────┘
```

### Partition Comparison (No Approval Needed)

| Partition | Max Time | GPUs | Use Case |
|-----------|----------|------|----------|
| short | 48 hours | 0 (CPU) | Setup, file transfers |
| gpu-interactive | 2 hours | 1 | Interactive GPU testing |
| gpu | **8 hours** | 1 | **Training jobs** ← USE THIS |

---

## Phase A: Initial Setup (One-Time)

### A1. Launch Interactive Session

1. Open browser: https://ood.explorer.northeastern.edu
2. Log in with Northeastern credentials
3. Click **"Standard Apps"** → **"VSCode (Beta)"**
4. Fill in:
   - Partition: `short` (CPU is fine for setup)
   - Time: `2` hours
   - CPUs: `8`
   - Memory: `32` GB
5. Click **"Launch"** and wait for session
6. Click **"Connect to VS Code"**

**Alternative (more reliable)**: Use SSH from your Mac terminal:
```bash
ssh kamarthi.v@login.discovery.neu.edu
```

### A2. Clone Repository

```bash
cd ~/ondemand
git clone https://github.com/vignankamarthi/FFN-pytorch-Experiment-Subset.git
cd FFN-pytorch-Experiment-Subset
```

### A3. Install Python Dependencies (NO CONDA)

**IMPORTANT**: Conda doesn't work in the Apptainer container. Use pip3 with --user flag:

```bash
# Check Python version (should be 3.9+)
python3 --version

# Install PyTorch with CUDA support
pip3 install --user torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install video loading library
pip3 install --user av

# Install other deps (may need to skip version pins)
pip3 install --user tensorboard tqdm einops
```

### A4. Verify Installation

```bash
python3 -c "import torch; print(torch.__version__)"
python3 -c "import av; print('av OK')"
```

**Note**: CUDA won't show as available in the Apptainer container, but it WILL work when you submit sbatch jobs to the gpu partition.

---

## Phase B: Download Dataset

### B1. Download from Qualcomm (Recommended)

Download directly on the cluster (faster than transferring from local):

1. Go to: https://www.qualcomm.com/developer/software/something-something-v-2-dataset/downloads
2. Copy download URLs (right-click → Copy Link)
3. Use wget on the cluster:

```bash
cd ~/ondemand/FFN-pytorch-Experiment-Subset
mkdir -p downloads
cd downloads

# Download video archives (replace with actual URLs from Qualcomm)
wget "https://qualcomm-url/20bn-something-something-v2-00.zip"
wget "https://qualcomm-url/20bn-something-something-v2-01.zip"

# Download labels
wget "https://qualcomm-url/something-something-v2-labels.zip"
```

**Alternative**: Transfer from your Mac (slower):
```bash
# From your Mac terminal (single line!)
scp ~/Downloads/*.zip kamarthi.v@login.discovery.neu.edu:~/ondemand/FFN-pytorch-Experiment-Subset/downloads/
```

### B2. Extract Video Archives

The video dataset is split across multiple archives that must be concatenated:

```bash
cd ~/ondemand/FFN-pytorch-Experiment-Subset/downloads

# Unzip the split archives (creates files like 20bn-something-something-v2-00, -01, etc.)
unzip 20bn-something-something-v2-00.zip
unzip 20bn-something-something-v2-01.zip

# Concatenate and extract (note the glob pattern)
cat 20bn-something-something-v2-0* | tar -xzf - -C ../database/data/

# The videos extract into a nested folder. Move them up:
cd ../database/data/20bn-something-something-v2
# Use find to avoid "argument list too long" error (220k+ files)
find . -name "*.webm" -exec mv {} ../20bn-something-something-v2/ \;

# Or if you're in the parent:
cd ..
find 20bn-something-something-v2/20bn-something-something-v2 -name "*.webm" -exec mv {} 20bn-something-something-v2/ \;
rmdir 20bn-something-something-v2/20bn-something-something-v2 2>/dev/null
```

### B3. Extract Labels

```bash
cd ~/ondemand/FFN-pytorch-Experiment-Subset/database/labels

# Unzip labels archive
unzip ~/ondemand/FFN-pytorch-Experiment-Subset/downloads/*labels*.zip

# If labels extract into a nested folder:
mv labels/* . 2>/dev/null || true
rmdir labels 2>/dev/null || true
```

### B4. Verify Dataset

```bash
cd ~/ondemand/FFN-pytorch-Experiment-Subset

# Count video files
ls database/data/20bn-something-something-v2/*.webm | wc -l
# Expected: 220847

# Check labels exist
ls database/labels/
# Expected: train.json, validation.json, labels.json, test.json
```

### B5. Clean Up Downloads

```bash
rm -rf ~/ondemand/FFN-pytorch-Experiment-Subset/downloads/
```

---

## Phase C: Verify Everything Works

### C1. Quick Sanity Check

Since tests require the full dataset loaded, skip pytest on cluster. Instead, verify Python/PyTorch work:

```bash
cd ~/ondemand/FFN-pytorch-Experiment-Subset

# Check Python and PyTorch
python3 -c "import torch; print(f'PyTorch {torch.__version__}')"
python3 -c "import av; print('PyAV OK')"

# Note: CUDA will show False in Apptainer container but WILL work in sbatch jobs
```

### C2. Verify Data Loading (Optional)

```bash
# Quick test that data loads (will fail without GPU, but verifies paths)
python3 -c "
from src.data import SomethingV2Dataset
ds = SomethingV2Dataset('database/data/20bn-something-something-v2', 'database/labels', split='train', num_frames=8)
print(f'Dataset size: {len(ds)}')
"
```

Expected output: `Dataset size: 168913`

---

## Phase D: Run Training (Job Chaining)

Submit Slurm batch jobs to the `gpu` partition. Jobs run independently of your VSCode session.

### D1. Slurm Scripts

Two scripts are included (updated to use pip packages, not conda):

**scripts/job_phase5_tsm.sbatch** - Vanilla TSM training (fits in one 8-hour job)

**scripts/job_phase7_ffn.sbatch** - FFN training (adaptive, resubmit until done)

The FFN script is **adaptive**: it automatically detects if `latest.pth` exists and resumes from it.

### D2. Submit Phase 5 (Vanilla TSM)

```bash
cd ~/ondemand/FFN-pytorch-Experiment-Subset
mkdir -p logs
sbatch scripts/job_phase5_tsm.sbatch
```

You'll see: `Submitted batch job 12345678`

### D3. Monitor Jobs

```bash
# Check job status
squeue -u $USER

# Watch output once job starts (replace JOB_ID)
tail -f logs/phase5_tsm_JOB_ID.out

# Check today's completed jobs
sacct -u $USER --starttime=today
```

Phase 5 should complete in ~4-6 hours.

### D4. After Phase 5 Completes

```bash
# Verify checkpoint was saved
ls -la checkpoints/vanilla_tsm/

# Submit FFN training
sbatch scripts/job_phase7_ffn.sbatch
```

### D5. FFN Job Chaining

FFN takes ~10-15 hours (2 job submissions):

```bash
# Check if FFN finished all 50 epochs
grep "Epoch \[50/50\]" logs/phase7_ffn_*.out

# If not done, resubmit (auto-resumes from checkpoint)
sbatch scripts/job_phase7_ffn.sbatch
```

Repeat until all 50 epochs complete.

### D3. Job Status Commands

```bash
squeue -u $USER              # Your running/pending jobs
sacct -u $USER --starttime=today  # Today's completed jobs
scancel <job_id>             # Cancel a job
```

### D4. Verify Checkpoints

After each job, verify checkpoints exist:
```bash
ls -la checkpoints/vanilla_tsm/
ls -la checkpoints/ffn/
```

---

## Phase E: Get Results Back

### E1. Commit to GitHub (from cluster)

```bash
cd ~/ondemand/FFN-pytorch-Experiment-Subset

git add checkpoints/ logs/
git commit -m "Training results from cluster

- Vanilla TSM: 50 epochs
- FFN: 50 epochs
- Cluster: Northeastern RC, gpu partition

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"

git push origin main
```

### E2. Pull to Local Mac

```bash
cd ~/Desktop/Work/Personal\ Projects/2025-2026/SMILE\ Lab\ Work/FFN-pytorch-Experiment-Subset
git pull origin main
```

---

## Troubleshooting

### "Job pending forever"

The gpu partition is shared. Peak times have longer queues.
```bash
squeue -p gpu  # Check partition queue
```

### "CUDA out of memory"

Reduce batch size in the sbatch script:
```bash
--batch_size 4
```

### "Job killed at 8 hours"

This is expected! Check that checkpoint was saved:
```bash
ls checkpoints/ffn/
```
Then resubmit the same script - it auto-resumes.

### "Cannot find checkpoint to resume"

Make sure the checkpoint path matches:
```bash
ls checkpoints/ffn/latest.pth
# Or find the epoch checkpoint:
ls checkpoints/ffn/epoch_*.pth
```

### "Module not found" or "No module named torch"

The sbatch environment uses pip --user packages. Verify:
```bash
export PATH="$HOME/.local/bin:$PATH"
python3 -c "import torch; print(torch.__version__)"
```

If packages missing, reinstall:
```bash
pip3 install --user torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip3 install --user av tensorboard tqdm einops
```

### "CUDA not available" in VSCode terminal

This is expected in the Apptainer container. CUDA **will** work when you submit sbatch jobs to the gpu partition. Don't worry about it during setup.

### Terminal character corruption / lag

The OOD VSCode terminal can be buggy. Options:
1. Open a new terminal tab (Terminal → New Terminal)
2. SSH directly: `ssh kamarthi.v@login.discovery.neu.edu`
3. Type `reset` to fix corrupted display

### "Argument list too long" when moving files

Use `find` with `-exec` for large numbers of files:
```bash
find source_dir -name "*.webm" -exec mv {} target_dir/ \;
```

---

## Checklist Summary

Before submitting training jobs:

- [ ] Repository cloned to `~/ondemand/FFN-pytorch-Experiment-Subset`
- [ ] PyTorch installed: `pip3 install --user torch torchvision`
- [ ] PyAV installed: `pip3 install --user av`
- [ ] Dataset extracted: 220,847 videos in `database/data/20bn-something-something-v2/`
- [ ] Labels present: `train.json`, `validation.json`, `labels.json` in `database/labels/`
- [ ] Python/PyTorch work: `python3 -c "import torch; print(torch.__version__)"`
- [ ] `logs/` directory created: `mkdir -p logs`

**Then submit jobs:**
```bash
sbatch scripts/job_phase5_tsm.sbatch    # Wait for completion
sbatch scripts/job_phase7_ffn.sbatch    # Resubmit until 50 epochs done
```

---

## Expected Results

### Vanilla TSM (TFD Demonstration)
| Eval Frames | Expected Accuracy |
|-------------|-------------------|
| 16F | ~61% |
| 8F | ~52% |
| 4F | ~31% |

**TFD Gap**: 30 percentage points (61% → 31%)

### FFN (TFD Recovery)
| Eval Frames | Expected Accuracy |
|-------------|-------------------|
| 16F | ~64% |
| 8F | ~62% |
| 4F | ~56% |

**FFN recovers 4F accuracy from ~31% to ~56%!**

---

*Last updated: February 2026*
