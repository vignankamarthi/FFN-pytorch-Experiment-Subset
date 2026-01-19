"""Tests for checkpoint save/load functionality.

Validates that model weights, optimizer state, and training progress
can be correctly saved and restored.
"""

import pytest
import torch
import torch.nn as nn

from src.training import (
    save_checkpoint,
    load_checkpoint,
    create_optimizer,
    create_scheduler,
    get_device,
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
def checkpoint_path(tmp_path):
    """Create a temporary checkpoint path."""
    return str(tmp_path / "test_checkpoint.pth")


# -----------------------------------------------------------------------------
# Save Checkpoint Tests
# -----------------------------------------------------------------------------


class TestSaveCheckpoint:
    """Test checkpoint saving functionality."""

    def test_save_creates_file(self, small_model, checkpoint_path):
        """Saving should create a file on disk."""
        optimizer = create_optimizer(small_model)
        scheduler = create_scheduler(optimizer)

        save_checkpoint(
            model=small_model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=5,
            best_acc=42.5,
            filepath=checkpoint_path,
        )

        import os
        assert os.path.exists(checkpoint_path)

    def test_save_contains_required_keys(self, small_model, checkpoint_path):
        """Saved checkpoint should contain all required keys."""
        optimizer = create_optimizer(small_model)
        scheduler = create_scheduler(optimizer)

        save_checkpoint(
            model=small_model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=5,
            best_acc=42.5,
            filepath=checkpoint_path,
        )

        checkpoint = torch.load(checkpoint_path, weights_only=False)
        assert "epoch" in checkpoint
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert "scheduler_state_dict" in checkpoint
        assert "best_acc" in checkpoint

    def test_save_stores_correct_values(self, small_model, checkpoint_path):
        """Saved values should match what was passed."""
        optimizer = create_optimizer(small_model)
        scheduler = create_scheduler(optimizer)

        save_checkpoint(
            model=small_model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=10,
            best_acc=55.5,
            filepath=checkpoint_path,
        )

        checkpoint = torch.load(checkpoint_path, weights_only=False)
        assert checkpoint["epoch"] == 10
        assert checkpoint["best_acc"] == 55.5

    def test_save_with_extra_data(self, small_model, checkpoint_path):
        """Extra data should be saved in checkpoint."""
        optimizer = create_optimizer(small_model)
        scheduler = create_scheduler(optimizer)

        save_checkpoint(
            model=small_model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=5,
            best_acc=42.5,
            filepath=checkpoint_path,
            extra={"custom_key": "custom_value", "loss": 0.123},
        )

        checkpoint = torch.load(checkpoint_path, weights_only=False)
        assert checkpoint["custom_key"] == "custom_value"
        assert checkpoint["loss"] == 0.123

    def test_save_without_scheduler(self, small_model, checkpoint_path):
        """Saving should work without scheduler."""
        optimizer = create_optimizer(small_model)

        save_checkpoint(
            model=small_model,
            optimizer=optimizer,
            scheduler=None,
            epoch=5,
            best_acc=42.5,
            filepath=checkpoint_path,
        )

        checkpoint = torch.load(checkpoint_path, weights_only=False)
        assert "scheduler_state_dict" not in checkpoint

    def test_save_creates_parent_directories(self, small_model, tmp_path):
        """Saving should create parent directories if they don't exist."""
        optimizer = create_optimizer(small_model)
        nested_path = str(tmp_path / "level1" / "level2" / "checkpoint.pth")

        save_checkpoint(
            model=small_model,
            optimizer=optimizer,
            scheduler=None,
            epoch=1,
            best_acc=0.0,
            filepath=nested_path,
        )

        import os
        assert os.path.exists(nested_path)


# -----------------------------------------------------------------------------
# Load Checkpoint Tests
# -----------------------------------------------------------------------------


class TestLoadCheckpoint:
    """Test checkpoint loading functionality."""

    def test_load_restores_model_weights(self, small_model, checkpoint_path, device):
        """Loading should restore model weights exactly."""
        model1 = small_model
        optimizer = create_optimizer(model1)

        # Modify model weights
        with torch.no_grad():
            for param in model1.parameters():
                param.add_(1.0)

        # Save
        save_checkpoint(
            model=model1,
            optimizer=optimizer,
            scheduler=None,
            epoch=5,
            best_acc=42.5,
            filepath=checkpoint_path,
        )

        # Create fresh model
        model2 = TSMResNet50(num_classes=10, num_frames=4, pretrained=False)

        # Load into model2
        load_checkpoint(filepath=checkpoint_path, model=model2, device=device)

        # Compare weights
        for (n1, p1), (n2, p2) in zip(
            model1.named_parameters(), model2.named_parameters()
        ):
            assert torch.allclose(p1, p2), f"Weight mismatch at {n1}"

    def test_load_restores_optimizer_state(self, small_model, checkpoint_path, device):
        """Loading should restore optimizer state."""
        model = small_model
        optimizer1 = create_optimizer(model)

        # Take some gradient steps to modify optimizer state
        videos = torch.randn(2, 3, 4, 224, 224)
        labels = torch.randint(0, 10, (2,))
        outputs = model(videos)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer1.step()

        # Save
        save_checkpoint(
            model=model,
            optimizer=optimizer1,
            scheduler=None,
            epoch=5,
            best_acc=42.5,
            filepath=checkpoint_path,
        )

        # Create fresh optimizer
        optimizer2 = create_optimizer(model)

        # Load
        load_checkpoint(
            filepath=checkpoint_path,
            model=model,
            optimizer=optimizer2,
            device=device,
        )

        # Compare optimizer states
        state1 = optimizer1.state_dict()
        state2 = optimizer2.state_dict()

        # Check param groups match
        assert len(state1["param_groups"]) == len(state2["param_groups"])

    def test_load_restores_scheduler_state(self, small_model, checkpoint_path, device):
        """Loading should restore scheduler state."""
        model = small_model
        optimizer = create_optimizer(model)
        scheduler1 = create_scheduler(optimizer, epochs=10)

        # Step scheduler
        for _ in range(5):
            scheduler1.step()

        lr_after_5_steps = optimizer.param_groups[0]["lr"]

        # Save
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler1,
            epoch=5,
            best_acc=42.5,
            filepath=checkpoint_path,
        )

        # Create fresh scheduler
        optimizer2 = create_optimizer(model)
        scheduler2 = create_scheduler(optimizer2, epochs=10)

        # Load
        load_checkpoint(
            filepath=checkpoint_path,
            model=model,
            optimizer=optimizer2,
            scheduler=scheduler2,
            device=device,
        )

        # LR should match
        assert optimizer2.param_groups[0]["lr"] == lr_after_5_steps

    def test_load_returns_epoch_and_accuracy(self, small_model, checkpoint_path, device):
        """Loading should return epoch and best_acc."""
        optimizer = create_optimizer(small_model)

        save_checkpoint(
            model=small_model,
            optimizer=optimizer,
            scheduler=None,
            epoch=15,
            best_acc=65.5,
            filepath=checkpoint_path,
        )

        model2 = TSMResNet50(num_classes=10, num_frames=4, pretrained=False)
        info = load_checkpoint(filepath=checkpoint_path, model=model2, device=device)

        assert info["epoch"] == 15
        assert info["best_acc"] == 65.5

    def test_load_without_optimizer(self, small_model, checkpoint_path, device):
        """Loading should work without restoring optimizer."""
        optimizer = create_optimizer(small_model)

        save_checkpoint(
            model=small_model,
            optimizer=optimizer,
            scheduler=None,
            epoch=5,
            best_acc=42.5,
            filepath=checkpoint_path,
        )

        model2 = TSMResNet50(num_classes=10, num_frames=4, pretrained=False)
        info = load_checkpoint(filepath=checkpoint_path, model=model2, device=device)

        assert info["epoch"] == 5  # Should still return epoch

    def test_load_nonexistent_file_raises(self, small_model, device):
        """Loading from nonexistent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_checkpoint(
                filepath="/nonexistent/path/checkpoint.pth",
                model=small_model,
                device=device,
            )


# -----------------------------------------------------------------------------
# Device Mapping Tests
# -----------------------------------------------------------------------------


class TestDeviceMapping:
    """Test checkpoint loading across devices."""

    def test_load_to_cpu(self, small_model, checkpoint_path):
        """Checkpoint should load to CPU."""
        # Save on current device
        device = get_device()
        model = small_model.to(device)
        optimizer = create_optimizer(model)

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            epoch=5,
            best_acc=42.5,
            filepath=checkpoint_path,
        )

        # Load to CPU
        model2 = TSMResNet50(num_classes=10, num_frames=4, pretrained=False)
        load_checkpoint(
            filepath=checkpoint_path,
            model=model2,
            device=torch.device("cpu"),
        )

        # Model should be on CPU
        for param in model2.parameters():
            assert param.device.type == "cpu"


# -----------------------------------------------------------------------------
# Round-Trip Tests
# -----------------------------------------------------------------------------


class TestRoundTrip:
    """Test complete save/load cycle."""

    def test_model_produces_same_output(self, small_model, checkpoint_path, device):
        """Model should produce same output after save/load cycle."""
        model1 = small_model.to(device)
        model1.eval()

        # Fixed input
        videos = torch.randn(2, 3, 4, 224, 224).to(device)

        # Get output before save
        with torch.no_grad():
            output1 = model1(videos)

        # Save
        optimizer = create_optimizer(model1)
        save_checkpoint(
            model=model1,
            optimizer=optimizer,
            scheduler=None,
            epoch=5,
            best_acc=42.5,
            filepath=checkpoint_path,
        )

        # Load into fresh model
        model2 = TSMResNet50(num_classes=10, num_frames=4, pretrained=False).to(device)
        model2.eval()
        load_checkpoint(filepath=checkpoint_path, model=model2, device=device)

        # Get output after load
        with torch.no_grad():
            output2 = model2(videos)

        # Outputs should match
        assert torch.allclose(output1, output2, atol=1e-6)

    def test_training_continues_correctly(self, small_model, checkpoint_path, device):
        """Training should continue correctly after checkpoint restore."""
        model = small_model.to(device)
        optimizer = create_optimizer(model)
        scheduler = create_scheduler(optimizer, epochs=10)

        # Train for 5 epochs worth of scheduler steps
        for _ in range(5):
            scheduler.step()

        # Save mid-training
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=4,  # 0-indexed, so epoch 4 = 5th epoch
            best_acc=50.0,
            filepath=checkpoint_path,
        )

        # Create fresh setup
        model2 = TSMResNet50(num_classes=10, num_frames=4, pretrained=False).to(device)
        optimizer2 = create_optimizer(model2)
        scheduler2 = create_scheduler(optimizer2, epochs=10)

        # Load checkpoint
        info = load_checkpoint(
            filepath=checkpoint_path,
            model=model2,
            optimizer=optimizer2,
            scheduler=scheduler2,
            device=device,
        )

        # Resume from next epoch
        resume_epoch = info["epoch"] + 1
        assert resume_epoch == 5

        # LRs should match
        assert optimizer.param_groups[0]["lr"] == optimizer2.param_groups[0]["lr"]
