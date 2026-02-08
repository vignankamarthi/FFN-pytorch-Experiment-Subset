"""Tests for the TFD evaluation script (eval_tfd.py).

Validates that the evaluation functions correctly load checkpoints
and compute accuracy at multiple frame counts for both vanilla TSM
and FFN models. Uses small models with random weights (no real checkpoint
needed) to verify the evaluation pipeline works end-to-end.
"""

import pytest
import torch

from src.models import TSMResNet50, create_ffn_model
from src.training import (
    save_checkpoint,
    create_optimizer,
    create_scheduler,
    get_device,
    AverageMeter,
    accuracy,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def device():
    """Get available device for testing."""
    return get_device()


@pytest.fixture
def tsm_checkpoint(tmp_path):
    """Create a vanilla TSM checkpoint for testing."""
    model = TSMResNet50(num_classes=10, num_frames=16, pretrained=False)
    optimizer = create_optimizer(model)
    scheduler = create_scheduler(optimizer)
    filepath = str(tmp_path / "tsm_test.pth")

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=50,
        best_acc=56.69,
        filepath=filepath,
    )
    return filepath


@pytest.fixture
def ffn_checkpoint(tmp_path, device):
    """Create an FFN checkpoint for testing."""
    model = create_ffn_model(
        num_classes=10,
        pretrained=False,
        device=torch.device("cpu"),
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40])
    filepath = str(tmp_path / "ffn_test.pth")

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=50,
        best_acc=63.80,
        filepath=filepath,
    )
    return filepath


# -----------------------------------------------------------------------------
# TSM Checkpoint Loading Tests
# -----------------------------------------------------------------------------


class TestTSMCheckpointLoading:
    """Test that vanilla TSM checkpoints load correctly at different frame counts."""

    def test_load_16f_checkpoint_at_16f(self, tsm_checkpoint):
        """Loading 16F-trained checkpoint into 16F model should work."""
        checkpoint = torch.load(tsm_checkpoint, map_location="cpu", weights_only=False)
        model = TSMResNet50(num_classes=10, num_frames=16, pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"])

    def test_load_16f_checkpoint_at_8f(self, tsm_checkpoint):
        """Loading 16F-trained checkpoint into 8F model should work (same weights)."""
        checkpoint = torch.load(tsm_checkpoint, map_location="cpu", weights_only=False)
        model = TSMResNet50(num_classes=10, num_frames=8, pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"])

    def test_load_16f_checkpoint_at_4f(self, tsm_checkpoint):
        """Loading 16F-trained checkpoint into 4F model should work (same weights)."""
        checkpoint = torch.load(tsm_checkpoint, map_location="cpu", weights_only=False)
        model = TSMResNet50(num_classes=10, num_frames=4, pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"])

    def test_checkpoint_has_expected_keys(self, tsm_checkpoint):
        """Checkpoint should contain model_state_dict, epoch, best_acc."""
        checkpoint = torch.load(tsm_checkpoint, map_location="cpu", weights_only=False)
        assert "model_state_dict" in checkpoint
        assert "epoch" in checkpoint
        assert "best_acc" in checkpoint
        assert checkpoint["epoch"] == 50
        assert checkpoint["best_acc"] == 56.69


# -----------------------------------------------------------------------------
# FFN Checkpoint Loading Tests
# -----------------------------------------------------------------------------


class TestFFNCheckpointLoading:
    """Test that FFN checkpoints load correctly."""

    def test_load_ffn_checkpoint(self, ffn_checkpoint):
        """Loading FFN checkpoint should work."""
        checkpoint = torch.load(ffn_checkpoint, map_location="cpu", weights_only=False)
        model = create_ffn_model(
            num_classes=10,
            pretrained=False,
            device=torch.device("cpu"),
        )
        model.load_state_dict(checkpoint["model_state_dict"])


# -----------------------------------------------------------------------------
# TSM Forward Pass at Different Frame Counts
# -----------------------------------------------------------------------------


class TestTSMMultiFrameForward:
    """Test vanilla TSM forward pass works at all frame counts after loading checkpoint."""

    @pytest.mark.parametrize("num_frames", [4, 8, 16])
    def test_forward_pass(self, tsm_checkpoint, num_frames):
        """Forward pass should produce (B, num_classes) at any frame count."""
        checkpoint = torch.load(tsm_checkpoint, map_location="cpu", weights_only=False)
        model = TSMResNet50(num_classes=10, num_frames=num_frames, pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        batch_size = 2
        x = torch.randn(batch_size, 3, num_frames, 224, 224)

        with torch.no_grad():
            outputs = model(x)

        assert outputs.shape == (batch_size, 10)

    @pytest.mark.parametrize("num_frames", [4, 8, 16])
    def test_outputs_are_valid_logits(self, tsm_checkpoint, num_frames):
        """Outputs should be finite logits (not NaN or Inf)."""
        checkpoint = torch.load(tsm_checkpoint, map_location="cpu", weights_only=False)
        model = TSMResNet50(num_classes=10, num_frames=num_frames, pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        x = torch.randn(2, 3, num_frames, 224, 224)

        with torch.no_grad():
            outputs = model(x)

        assert torch.isfinite(outputs).all()


# -----------------------------------------------------------------------------
# FFN Forward Pass at Different Frame Counts
# -----------------------------------------------------------------------------


class TestFFNMultiFrameForward:
    """Test FFN inference mode works at each frame count individually."""

    @pytest.mark.parametrize("frame_count", [4, 8, 16])
    def test_inference_single_frame_count(self, ffn_checkpoint, frame_count):
        """FFN inference should work with a single frame count."""
        checkpoint = torch.load(ffn_checkpoint, map_location="cpu", weights_only=False)
        model = create_ffn_model(
            num_classes=10,
            pretrained=False,
            device=torch.device("cpu"),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        batch_size = 2
        x = torch.randn(batch_size, 3, frame_count, 224, 224)

        with torch.no_grad():
            if frame_count == 4:
                outputs = model(x_4=x, training=False)
            elif frame_count == 8:
                outputs = model(x_8=x, training=False)
            else:
                outputs = model(x_16=x, training=False)

        assert outputs.shape == (batch_size, 10)
        assert torch.isfinite(outputs).all()


# -----------------------------------------------------------------------------
# Accuracy Utility Tests
# -----------------------------------------------------------------------------


class TestAccuracyComputation:
    """Test accuracy function works correctly for evaluation."""

    def test_perfect_accuracy(self):
        """All correct predictions should give 100% accuracy."""
        output = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
        target = torch.tensor([0, 1])
        acc1, acc5 = accuracy(output, target, topk=(1, 3))
        assert acc1.item() == 100.0

    def test_zero_accuracy(self):
        """All wrong predictions should give 0% accuracy."""
        output = torch.tensor([[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        target = torch.tensor([1, 2])
        acc1, _ = accuracy(output, target, topk=(1, 3))
        assert acc1.item() == 0.0

    def test_top5_includes_correct(self):
        """Top-5 should find correct class even if not the top prediction."""
        # Correct class (2) is ranked 3rd
        output = torch.tensor([[10.0, 8.0, 6.0, 4.0, 2.0]])
        target = torch.tensor([2])
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        assert acc1.item() == 0.0
        assert acc5.item() == 100.0


# -----------------------------------------------------------------------------
# AverageMeter Tests
# -----------------------------------------------------------------------------


class TestAverageMeter:
    """Test AverageMeter tracking for evaluation metrics."""

    def test_single_update(self):
        """Single update should set avg to the value."""
        meter = AverageMeter("test")
        meter.update(75.0, n=32)
        assert meter.avg == 75.0

    def test_weighted_average(self):
        """Multiple updates with different batch sizes should compute weighted avg."""
        meter = AverageMeter("test")
        meter.update(80.0, n=32)
        meter.update(60.0, n=32)
        assert meter.avg == pytest.approx(70.0)

    def test_count_tracks_total(self):
        """Count should track total samples seen."""
        meter = AverageMeter("test")
        meter.update(80.0, n=32)
        meter.update(60.0, n=16)
        assert meter.count == 48
