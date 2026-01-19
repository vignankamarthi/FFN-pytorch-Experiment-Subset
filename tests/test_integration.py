"""Integration tests for data + model + training pipeline.

Tests the complete flow from video loading through model inference
and loss computation. Catches issues that unit tests might miss.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.data import SSv2Dataset, get_train_transforms, get_val_transforms
from src.models import TSMResNet50
from src.training import get_device, create_optimizer, accuracy


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def device():
    """Get available device for testing."""
    return get_device()


@pytest.fixture
def model(device):
    """Create TSM model on device."""
    model = TSMResNet50(num_classes=174, num_frames=16, pretrained=False)
    return model.to(device)


@pytest.fixture
def synthetic_video_batch(device):
    """Create synthetic video batch matching SSv2 format."""
    # Shape: (B, C, T, H, W) = (4, 3, 16, 224, 224)
    videos = torch.randn(4, 3, 16, 224, 224).to(device)
    labels = torch.randint(0, 174, (4,)).to(device)
    return videos, labels


# -----------------------------------------------------------------------------
# Data-Model Interface Tests
# -----------------------------------------------------------------------------


class TestDataModelInterface:
    """Test that data output matches model input expectations."""

    def test_synthetic_video_shape(self, model, synthetic_video_batch, device):
        """Synthetic videos should have correct shape for model."""
        videos, labels = synthetic_video_batch

        # Check input shape
        assert videos.shape == (4, 3, 16, 224, 224)
        assert labels.shape == (4,)

        # Model should accept this input
        model.eval()
        with torch.no_grad():
            outputs = model(videos)

        assert outputs.shape == (4, 174)

    def test_different_frame_counts(self, device):
        """Model should work with different frame counts."""
        for num_frames in [4, 8, 16]:
            model = TSMResNet50(
                num_classes=174, num_frames=num_frames, pretrained=False
            ).to(device)

            videos = torch.randn(2, 3, num_frames, 224, 224).to(device)

            model.eval()
            with torch.no_grad():
                outputs = model(videos)

            assert outputs.shape == (2, 174), f"Failed for {num_frames} frames"


# -----------------------------------------------------------------------------
# Loss Computation Tests
# -----------------------------------------------------------------------------


class TestLossComputation:
    """Test loss computation in training pipeline."""

    def test_cross_entropy_loss(self, model, synthetic_video_batch, device):
        """CrossEntropyLoss should work with model outputs."""
        videos, labels = synthetic_video_batch
        criterion = nn.CrossEntropyLoss()

        outputs = model(videos)
        loss = criterion(outputs, labels)

        # Loss should be positive and finite
        assert loss.item() > 0
        assert torch.isfinite(loss)

    def test_loss_is_differentiable(self, model, synthetic_video_batch, device):
        """Loss should have gradients that flow back to model."""
        videos, labels = synthetic_video_batch
        criterion = nn.CrossEntropyLoss()

        outputs = model(videos)
        loss = criterion(outputs, labels)
        loss.backward()

        # At least some parameters should have gradients
        grad_count = sum(
            1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0
        )
        assert grad_count > 0, "No gradients computed"

    def test_accuracy_computation(self, model, synthetic_video_batch, device):
        """Accuracy should be computable from outputs."""
        videos, labels = synthetic_video_batch

        model.eval()
        with torch.no_grad():
            outputs = model(videos)
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))

        # Accuracy should be between 0 and 100
        assert 0 <= acc1.item() <= 100
        assert 0 <= acc5.item() <= 100


# -----------------------------------------------------------------------------
# Training Step Integration Tests
# -----------------------------------------------------------------------------


class TestTrainingStepIntegration:
    """Test complete training step with all components."""

    def test_single_training_step(self, model, synthetic_video_batch, device):
        """Complete training step should work end-to-end."""
        videos, labels = synthetic_video_batch
        optimizer = create_optimizer(model)
        criterion = nn.CrossEntropyLoss()

        # Forward
        outputs = model(videos)
        loss = criterion(outputs, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Should complete without error
        assert True

    def test_multiple_training_steps(self, model, device):
        """Multiple training steps should work correctly."""
        optimizer = create_optimizer(model)
        criterion = nn.CrossEntropyLoss()

        losses = []
        for _ in range(5):
            videos = torch.randn(2, 3, 16, 224, 224).to(device)
            labels = torch.randint(0, 174, (2,)).to(device)

            outputs = model(videos)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # All losses should be finite
        assert all(l > 0 and l < 100 for l in losses)

    def test_train_eval_mode_switching(self, model, synthetic_video_batch, device):
        """Model should switch between train and eval modes correctly."""
        videos, labels = synthetic_video_batch

        # Train mode
        model.train()
        outputs_train = model(videos)
        assert outputs_train.shape == (4, 174)

        # Eval mode
        model.eval()
        with torch.no_grad():
            outputs_eval = model(videos)
        assert outputs_eval.shape == (4, 174)

        # Outputs may differ due to dropout
        # Just verify both work


# -----------------------------------------------------------------------------
# Batch Size Tests
# -----------------------------------------------------------------------------


class TestBatchSizes:
    """Test different batch sizes work correctly."""

    def test_batch_size_1(self, device):
        """Model should work with batch size 1."""
        model = TSMResNet50(num_classes=174, num_frames=16, pretrained=False).to(device)
        model.eval()

        videos = torch.randn(1, 3, 16, 224, 224).to(device)

        with torch.no_grad():
            outputs = model(videos)

        assert outputs.shape == (1, 174)

    def test_batch_size_8(self, device):
        """Model should work with default batch size 8."""
        model = TSMResNet50(num_classes=174, num_frames=16, pretrained=False).to(device)
        model.eval()

        videos = torch.randn(8, 3, 16, 224, 224).to(device)

        with torch.no_grad():
            outputs = model(videos)

        assert outputs.shape == (8, 174)

    def test_odd_batch_size(self, device):
        """Model should work with odd batch sizes."""
        model = TSMResNet50(num_classes=174, num_frames=16, pretrained=False).to(device)
        model.eval()

        videos = torch.randn(7, 3, 16, 224, 224).to(device)

        with torch.no_grad():
            outputs = model(videos)

        assert outputs.shape == (7, 174)


# -----------------------------------------------------------------------------
# Memory and Performance Tests
# -----------------------------------------------------------------------------


class TestMemory:
    """Test memory-related behavior."""

    def test_gradient_accumulation(self, model, device):
        """Gradients should accumulate correctly."""
        optimizer = create_optimizer(model)
        criterion = nn.CrossEntropyLoss()

        # First step
        videos1 = torch.randn(2, 3, 16, 224, 224).to(device)
        labels1 = torch.randint(0, 174, (2,)).to(device)
        loss1 = criterion(model(videos1), labels1)
        loss1.backward()

        # Second step without zero_grad
        videos2 = torch.randn(2, 3, 16, 224, 224).to(device)
        labels2 = torch.randint(0, 174, (2,)).to(device)
        loss2 = criterion(model(videos2), labels2)
        loss2.backward()

        # Now step with accumulated gradients
        optimizer.step()
        optimizer.zero_grad()

        # Should complete without error
        assert True

    def test_no_grad_context(self, model, synthetic_video_batch, device):
        """no_grad context should prevent gradient computation."""
        videos, labels = synthetic_video_batch

        model.eval()
        with torch.no_grad():
            outputs = model(videos)
            loss = nn.CrossEntropyLoss()(outputs, labels)

        # Should not have grad_fn
        assert not loss.requires_grad


# -----------------------------------------------------------------------------
# DataLoader Integration (Synthetic)
# -----------------------------------------------------------------------------


class TestDataLoaderIntegration:
    """Test model with DataLoader (using synthetic data)."""

    def test_dataloader_batch_iteration(self, device):
        """Model should process batches from DataLoader."""
        model = TSMResNet50(num_classes=174, num_frames=16, pretrained=False).to(device)
        model.eval()

        # Create synthetic dataset
        videos = torch.randn(16, 3, 16, 224, 224)
        labels = torch.randint(0, 174, (16,))
        dataset = TensorDataset(videos, labels)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

        all_outputs = []
        with torch.no_grad():
            for batch_videos, batch_labels in dataloader:
                batch_videos = batch_videos.to(device)
                outputs = model(batch_videos)
                all_outputs.append(outputs)

        # Should process all batches
        assert len(all_outputs) == 4
        assert all(o.shape == (4, 174) for o in all_outputs)

    def test_training_loop_with_dataloader(self, device):
        """Complete training loop with DataLoader should work."""
        model = TSMResNet50(num_classes=10, num_frames=8, pretrained=False).to(device)
        optimizer = create_optimizer(model)
        criterion = nn.CrossEntropyLoss()

        # Create synthetic dataset
        videos = torch.randn(8, 3, 8, 224, 224)
        labels = torch.randint(0, 10, (8,))
        dataset = TensorDataset(videos, labels)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

        # Train for one epoch
        model.train()
        epoch_loss = 0.0
        for batch_videos, batch_labels in dataloader:
            batch_videos = batch_videos.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_videos)
            loss = criterion(outputs, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Should have positive loss
        assert epoch_loss > 0


# -----------------------------------------------------------------------------
# Edge Cases
# -----------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and potential failure modes."""

    def test_zero_input(self, model, device):
        """Model should handle zero input without NaN."""
        videos = torch.zeros(2, 3, 16, 224, 224).to(device)

        model.eval()
        with torch.no_grad():
            outputs = model(videos)

        assert torch.isfinite(outputs).all(), "Output should be finite for zero input"

    def test_large_input_values(self, model, device):
        """Model should handle large input values."""
        videos = torch.randn(2, 3, 16, 224, 224).to(device) * 10  # Large values

        model.eval()
        with torch.no_grad():
            outputs = model(videos)

        assert torch.isfinite(outputs).all(), "Output should be finite for large input"

    def test_normalized_input(self, model, device):
        """Model should work with ImageNet-normalized input."""
        # Create properly normalized input
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1).to(device)

        # Raw values in [0, 1]
        raw = torch.rand(2, 3, 16, 224, 224).to(device)
        # Normalize
        normalized = (raw - mean) / std

        model.eval()
        with torch.no_grad():
            outputs = model(normalized)

        assert outputs.shape == (2, 174)
        assert torch.isfinite(outputs).all()
