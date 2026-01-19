"""Tests for training infrastructure.

Validates training loop, optimizer, scheduler, and loss computation.
Uses synthetic data to keep tests fast while catching real bugs.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.training import (
    Trainer,
    create_optimizer,
    create_scheduler,
    get_device,
    AverageMeter,
    accuracy,
)
from src.models import TSMResNet50


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def device():
    """Get available device for testing."""
    return get_device()


@pytest.fixture
def small_model():
    """Create a small TSM model for fast testing."""
    return TSMResNet50(num_classes=10, num_frames=4, pretrained=False)


@pytest.fixture
def synthetic_dataloader():
    """Create synthetic data loader for training tests."""
    # Synthetic video data: 16 samples, 4 frames, 224x224
    videos = torch.randn(16, 3, 4, 224, 224)
    labels = torch.randint(0, 10, (16,))
    dataset = TensorDataset(videos, labels)
    return DataLoader(dataset, batch_size=4, shuffle=True)


# -----------------------------------------------------------------------------
# Device Selection Tests
# -----------------------------------------------------------------------------


class TestDeviceSelection:
    """Test device selection utility."""

    def test_get_device_returns_valid_device(self):
        """get_device should return a valid torch.device."""
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ["cuda", "mps", "cpu"]

    def test_device_is_available(self, device):
        """Returned device should be usable."""
        # Create tensor on device
        x = torch.randn(2, 2).to(device)
        assert x.device.type == device.type


# -----------------------------------------------------------------------------
# AverageMeter Tests
# -----------------------------------------------------------------------------


class TestAverageMeter:
    """Test metric tracking utility."""

    def test_initial_state(self):
        """Meter should start at zero."""
        meter = AverageMeter("test")
        assert meter.avg == 0.0
        assert meter.sum == 0.0
        assert meter.count == 0

    def test_single_update(self):
        """Single update should set val and avg correctly."""
        meter = AverageMeter("test")
        meter.update(5.0)
        assert meter.val == 5.0
        assert meter.avg == 5.0
        assert meter.count == 1

    def test_multiple_updates(self):
        """Multiple updates should compute correct average."""
        meter = AverageMeter("test")
        meter.update(2.0)
        meter.update(4.0)
        meter.update(6.0)
        assert meter.avg == 4.0
        assert meter.count == 3

    def test_weighted_update(self):
        """Weighted update should compute weighted average."""
        meter = AverageMeter("test")
        meter.update(2.0, n=2)  # 2 samples of value 2.0
        meter.update(4.0, n=2)  # 2 samples of value 4.0
        assert meter.avg == 3.0
        assert meter.count == 4

    def test_reset(self):
        """Reset should clear all values."""
        meter = AverageMeter("test")
        meter.update(5.0)
        meter.reset()
        assert meter.avg == 0.0
        assert meter.count == 0


# -----------------------------------------------------------------------------
# Accuracy Tests
# -----------------------------------------------------------------------------


class TestAccuracy:
    """Test accuracy computation."""

    def test_top1_perfect(self):
        """Perfect predictions should give 100% accuracy."""
        output = torch.tensor([
            [10.0, 0.0, 0.0],  # Predicts class 0
            [0.0, 10.0, 0.0],  # Predicts class 1
            [0.0, 0.0, 10.0],  # Predicts class 2
        ])
        target = torch.tensor([0, 1, 2])
        acc1, = accuracy(output, target, topk=(1,))
        assert acc1.item() == 100.0

    def test_top1_zero(self):
        """Completely wrong predictions should give 0% accuracy."""
        output = torch.tensor([
            [10.0, 0.0, 0.0],  # Predicts class 0
            [10.0, 0.0, 0.0],  # Predicts class 0
            [10.0, 0.0, 0.0],  # Predicts class 0
        ])
        target = torch.tensor([1, 2, 1])  # All wrong
        acc1, = accuracy(output, target, topk=(1,))
        assert acc1.item() == 0.0

    def test_top5_includes_correct(self):
        """Top-5 should find correct class in top 5 predictions."""
        # Output where true class is 5th highest
        output = torch.tensor([
            [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        ])
        target = torch.tensor([4])  # 5th highest score
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        assert acc1.item() == 0.0  # Not top-1
        assert acc5.item() == 100.0  # Is top-5

    def test_batch_accuracy(self):
        """Accuracy should work correctly over batches."""
        output = torch.tensor([
            [10.0, 0.0],  # Correct (class 0)
            [10.0, 0.0],  # Wrong (should be class 1)
        ])
        target = torch.tensor([0, 1])
        acc1, = accuracy(output, target, topk=(1,))
        assert acc1.item() == 50.0


# -----------------------------------------------------------------------------
# Optimizer Tests
# -----------------------------------------------------------------------------


class TestOptimizer:
    """Test optimizer creation."""

    def test_create_optimizer_default(self, small_model):
        """Create optimizer with default hyperparameters."""
        optimizer = create_optimizer(small_model)
        assert isinstance(optimizer, torch.optim.SGD)
        # Check default lr
        assert optimizer.defaults["lr"] == 0.01
        assert optimizer.defaults["momentum"] == 0.9
        assert optimizer.defaults["weight_decay"] == 5e-4

    def test_create_optimizer_custom_lr(self, small_model):
        """Create optimizer with custom learning rate."""
        optimizer = create_optimizer(small_model, lr=0.001)
        assert optimizer.defaults["lr"] == 0.001

    def test_optimizer_has_params(self, small_model):
        """Optimizer should have model parameters."""
        optimizer = create_optimizer(small_model)
        param_count = sum(len(g["params"]) for g in optimizer.param_groups)
        assert param_count > 0


# -----------------------------------------------------------------------------
# Scheduler Tests
# -----------------------------------------------------------------------------


class TestScheduler:
    """Test learning rate scheduler."""

    def test_create_scheduler(self, small_model):
        """Create scheduler with default settings."""
        optimizer = create_optimizer(small_model)
        scheduler = create_scheduler(optimizer, epochs=50)
        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_scheduler_decreases_lr(self, small_model):
        """LR should decrease over epochs."""
        optimizer = create_optimizer(small_model, lr=0.01)
        scheduler = create_scheduler(optimizer, epochs=10)

        initial_lr = optimizer.param_groups[0]["lr"]

        # Step through half the epochs
        for _ in range(5):
            scheduler.step()

        mid_lr = optimizer.param_groups[0]["lr"]
        assert mid_lr < initial_lr

    def test_scheduler_reaches_minimum(self, small_model):
        """LR should be near zero at end of training."""
        optimizer = create_optimizer(small_model, lr=0.01)
        scheduler = create_scheduler(optimizer, epochs=10)

        # Step through all epochs
        for _ in range(10):
            scheduler.step()

        final_lr = optimizer.param_groups[0]["lr"]
        assert final_lr < 0.001  # Should be very small


# -----------------------------------------------------------------------------
# Training Step Tests
# -----------------------------------------------------------------------------


class TestTrainingStep:
    """Test single training step mechanics."""

    def test_forward_backward_pass(self, small_model, device):
        """Model should complete forward and backward pass."""
        model = small_model.to(device)
        optimizer = create_optimizer(model)

        # Synthetic batch
        videos = torch.randn(2, 3, 4, 224, 224).to(device)
        labels = torch.randint(0, 10, (2,)).to(device)

        # Forward
        outputs = model(videos)
        loss = nn.CrossEntropyLoss()(outputs, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Verify gradients exist
        has_grads = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        assert has_grads

    def test_loss_is_finite(self, small_model, device):
        """Loss should be finite (not NaN or Inf)."""
        model = small_model.to(device)

        videos = torch.randn(2, 3, 4, 224, 224).to(device)
        labels = torch.randint(0, 10, (2,)).to(device)

        outputs = model(videos)
        loss = nn.CrossEntropyLoss()(outputs, labels)

        assert torch.isfinite(loss), "Loss should be finite"

    def test_gradients_are_finite(self, small_model, device):
        """All gradients should be finite."""
        model = small_model.to(device)

        videos = torch.randn(2, 3, 4, 224, 224).to(device)
        labels = torch.randint(0, 10, (2,)).to(device)

        outputs = model(videos)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), f"Gradient for {name} is not finite"


# -----------------------------------------------------------------------------
# Training Loop Tests
# -----------------------------------------------------------------------------


class TestTrainingLoop:
    """Test multi-step training behavior."""

    def test_loss_decreases(self, small_model, device):
        """Loss should decrease over multiple steps on same batch."""
        model = small_model.to(device)
        optimizer = create_optimizer(model, lr=0.01)

        # Fixed batch (overfitting test)
        videos = torch.randn(4, 3, 4, 224, 224).to(device)
        labels = torch.randint(0, 10, (4,)).to(device)
        criterion = nn.CrossEntropyLoss()

        losses = []
        for _ in range(10):
            outputs = model(videos)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # Loss should decrease (allow some noise)
        assert losses[-1] < losses[0], "Loss should decrease over training steps"

    def test_multiple_batches(self, small_model, synthetic_dataloader, device):
        """Training should work across multiple batches."""
        model = small_model.to(device)
        optimizer = create_optimizer(model)
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        batch_count = 0

        for videos, labels in synthetic_dataloader:
            videos = videos.to(device)
            labels = labels.to(device)

            outputs = model(videos)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        avg_loss = total_loss / batch_count
        assert avg_loss > 0, "Should have positive loss"
        assert batch_count == 4, "Should process all batches"


# -----------------------------------------------------------------------------
# Trainer Class Tests
# -----------------------------------------------------------------------------


class TestTrainer:
    """Test the Trainer class."""

    @pytest.fixture
    def trainer_setup(self, small_model, synthetic_dataloader, device, tmp_path):
        """Create trainer with all components."""
        model = small_model
        optimizer = create_optimizer(model)
        scheduler = create_scheduler(optimizer, epochs=5)

        trainer = Trainer(
            model=model,
            train_loader=synthetic_dataloader,
            val_loader=synthetic_dataloader,  # Use same for simplicity
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            checkpoint_dir=str(tmp_path / "checkpoints"),
        )
        return trainer

    def test_trainer_initialization(self, trainer_setup):
        """Trainer should initialize correctly."""
        trainer = trainer_setup
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None
        assert trainer.best_acc == 0.0

    def test_train_epoch(self, trainer_setup):
        """train_epoch should return metrics dict."""
        trainer = trainer_setup
        metrics = trainer.train_epoch()

        assert "loss" in metrics
        assert "top1_acc" in metrics
        assert "top5_acc" in metrics
        assert metrics["loss"] > 0

    def test_validate(self, trainer_setup):
        """validate should return metrics dict."""
        trainer = trainer_setup
        metrics = trainer.validate()

        assert "loss" in metrics
        assert "top1_acc" in metrics
        assert "top5_acc" in metrics

    def test_short_training(self, trainer_setup):
        """Full training loop should work for few epochs."""
        trainer = trainer_setup

        # Train for 2 epochs
        trainer.train(
            epochs=2,
            start_epoch=0,
            validate_every=1,
            save_every=1,
        )

        # Should have tracked best accuracy
        assert trainer.current_epoch == 1  # 0-indexed, so epoch 1 is the last


# -----------------------------------------------------------------------------
# Edge Cases
# -----------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_meter(self):
        """AverageMeter should handle no updates gracefully."""
        meter = AverageMeter("empty")
        assert meter.avg == 0.0  # Should not raise

    def test_single_sample_accuracy(self):
        """Accuracy should work with batch size 1."""
        output = torch.tensor([[10.0, 0.0]])
        target = torch.tensor([0])
        acc1, = accuracy(output, target, topk=(1,))
        assert acc1.item() == 100.0

    def test_cpu_training(self, small_model):
        """Training should work on CPU."""
        device = torch.device("cpu")
        model = small_model.to(device)
        optimizer = create_optimizer(model)

        videos = torch.randn(2, 3, 4, 224, 224)
        labels = torch.randint(0, 10, (2,))

        outputs = model(videos)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()

        # Should complete without error
        assert True
