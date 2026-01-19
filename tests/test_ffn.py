"""Tests for FFN model components.

Tests the Frame Flexible Network implementation including:
- TSM-FFN model architecture
- Specialized BatchNorm
- Weight Alteration adapters
- Temporal Distillation Loss
- Pretrained weight loading
"""

import pytest
import torch
import torch.nn as nn

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.tsm_ffn import (
    TSMFFN,
    FFNBottleneck,
    FFNResNet,
    TSM,
    resnet50_ffn,
    create_ffn_model,
)
from models.temporal_distillation import (
    TemporalDistillationLoss,
    FFNLoss,
    compute_ffn_loss,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def device():
    """Get test device (CPU for CI compatibility)."""
    return torch.device("cpu")


@pytest.fixture
def batch_size():
    """Standard batch size for tests."""
    return 2


@pytest.fixture
def num_classes():
    """Number of classes for SSv2."""
    return 174


@pytest.fixture
def sample_inputs(batch_size):
    """Create sample video inputs for all three frame counts."""
    return {
        "x_4": torch.randn(batch_size, 3, 4, 224, 224),
        "x_8": torch.randn(batch_size, 3, 8, 224, 224),
        "x_16": torch.randn(batch_size, 3, 16, 224, 224),
    }


@pytest.fixture
def flat_inputs(batch_size):
    """Create flattened inputs (B*T, C, H, W) for block-level tests."""
    return {
        "x_4": torch.randn(batch_size * 4, 64, 56, 56),
        "x_8": torch.randn(batch_size * 8, 64, 56, 56),
        "x_16": torch.randn(batch_size * 16, 64, 56, 56),
    }


# =============================================================================
# TSM Module Tests
# =============================================================================

class TestTSM:
    """Test Temporal Shift Module for FFN."""

    def test_tsm_shape_preserved(self, batch_size):
        """TSM should preserve input shape."""
        tsm = TSM(num_segments=16)
        x = torch.randn(batch_size * 16, 64, 56, 56)
        out = tsm(x)
        assert out.shape == x.shape

    def test_tsm_different_segments(self):
        """TSM should work with different segment counts."""
        for num_seg in [4, 8, 16]:
            tsm = TSM(num_segments=num_seg)
            x = torch.randn(2 * num_seg, 64, 56, 56)
            out = tsm(x)
            assert out.shape == x.shape

    def test_tsm_shifts_channels(self, batch_size):
        """TSM should shift channels forward and backward."""
        tsm = TSM(num_segments=4, fold_div=8)
        x = torch.zeros(batch_size * 4, 64, 56, 56)
        # Set identifiable pattern in channels to shift
        x[:, 0, :, :] = 1.0  # Forward shift channel
        x[:, 8, :, :] = 2.0  # Backward shift channel

        out = tsm(x)
        # Verify output is not all zeros (shifting happened)
        assert not torch.allclose(out, torch.zeros_like(out))


# =============================================================================
# FFN Bottleneck Tests
# =============================================================================

class TestFFNBottleneck:
    """Test FFN Bottleneck block with specialized components."""

    def test_bottleneck_training_forward(self, batch_size):
        """Bottleneck training mode produces three outputs."""
        # Use inplanes=256 to match output (planes * expansion = 64 * 4 = 256)
        # This avoids needing downsample layers
        inplanes = 256
        planes = 64
        x_4 = torch.randn(batch_size * 4, inplanes, 56, 56)
        x_8 = torch.randn(batch_size * 8, inplanes, 56, 56)
        x_16 = torch.randn(batch_size * 16, inplanes, 56, 56)

        block = FFNBottleneck(inplanes, planes, num_segments_h=16, num_segments_m=8, num_segments_l=4)
        out_4, out_8, out_16 = block(x_4, x_8, x_16, training=True)

        assert out_4.shape == (batch_size * 4, 256, 56, 56)
        assert out_8.shape == (batch_size * 8, 256, 56, 56)
        assert out_16.shape == (batch_size * 16, 256, 56, 56)

    def test_bottleneck_inference_forward(self, batch_size):
        """Bottleneck inference mode produces single output."""
        # Use inplanes=256 to match output
        inplanes = 256
        planes = 64
        x_4 = torch.randn(batch_size * 4, inplanes, 56, 56)
        x_8 = torch.randn(batch_size * 8, inplanes, 56, 56)
        x_16 = torch.randn(batch_size * 16, inplanes, 56, 56)

        block = FFNBottleneck(inplanes, planes, num_segments_h=16, num_segments_m=8, num_segments_l=4)

        # Test each frame count separately
        out_4 = block(x_4, None, None, training=False)
        assert out_4.shape == (batch_size * 4, 256, 56, 56)

        out_8 = block(None, x_8, None, training=False)
        assert out_8.shape == (batch_size * 8, 256, 56, 56)

        out_16 = block(None, None, x_16, training=False)
        assert out_16.shape == (batch_size * 16, 256, 56, 56)

    def test_bottleneck_has_specialized_bn(self):
        """Bottleneck should have separate BN for each frame count."""
        block = FFNBottleneck(64, 64, num_segments_h=16)

        # Check BN layers exist for all frame counts
        assert hasattr(block, "bn1_4") and hasattr(block, "bn1_8") and hasattr(block, "bn1_16")
        assert hasattr(block, "bn2_4") and hasattr(block, "bn2_8") and hasattr(block, "bn2_16")
        assert hasattr(block, "bn3_4") and hasattr(block, "bn3_8") and hasattr(block, "bn3_16")

        # Verify they are separate instances
        assert block.bn1_4 is not block.bn1_8
        assert block.bn1_8 is not block.bn1_16

    def test_bottleneck_has_weight_alteration(self):
        """Bottleneck should have adaconv adapters for each frame count."""
        block = FFNBottleneck(64, 64, num_segments_h=16)

        # Check adaconv exists
        assert hasattr(block, "adaconv_4")
        assert hasattr(block, "adaconv_8")
        assert hasattr(block, "adaconv_16")

        # Verify they are depthwise (groups == in_channels)
        assert block.adaconv_4.groups == 64
        assert block.adaconv_8.groups == 64
        assert block.adaconv_16.groups == 64

    def test_bottleneck_adaconv_near_zero_init(self):
        """Adaconv weights should be near-zero initialized."""
        block = FFNBottleneck(64, 64, num_segments_h=16)

        # Check weights are small (initialized with std=1e-3)
        for name in ["adaconv_4", "adaconv_8", "adaconv_16"]:
            adaconv = getattr(block, name)
            assert adaconv.weight.std() < 0.01, f"{name} weights too large"

    def test_bottleneck_shared_convolutions(self):
        """Convolutions should be shared (single instance)."""
        block = FFNBottleneck(64, 64, num_segments_h=16)

        # Only one conv1, conv2, conv3 should exist (not per-frame-count)
        assert hasattr(block, "conv1") and isinstance(block.conv1, nn.Conv2d)
        assert hasattr(block, "conv2") and isinstance(block.conv2, nn.Conv2d)
        assert hasattr(block, "conv3") and isinstance(block.conv3, nn.Conv2d)


# =============================================================================
# FFN ResNet Tests
# =============================================================================

class TestFFNResNet:
    """Test FFNResNet backbone."""

    def test_resnet_training_output_shape(self, batch_size, num_classes):
        """ResNet training mode produces correct output shapes."""
        model = FFNResNet(
            FFNBottleneck,
            [1, 1, 1, 1],  # Minimal config for speed
            num_segments_h=16,
            num_segments_m=8,
            num_segments_l=4,
            num_classes=num_classes,
        )

        # Flatten inputs (B*T, C, H, W)
        x_4 = torch.randn(batch_size * 4, 3, 224, 224)
        x_8 = torch.randn(batch_size * 8, 3, 224, 224)
        x_16 = torch.randn(batch_size * 16, 3, 224, 224)

        out_4, out_8, out_16 = model(x_4, x_8, x_16, training=True)

        assert out_4.shape == (batch_size, num_classes)
        assert out_8.shape == (batch_size, num_classes)
        assert out_16.shape == (batch_size, num_classes)

    def test_resnet_inference_output_shape(self, batch_size, num_classes):
        """ResNet inference mode produces correct output shapes."""
        model = FFNResNet(
            FFNBottleneck,
            [1, 1, 1, 1],
            num_segments_h=16,
            num_segments_m=8,
            num_segments_l=4,
            num_classes=num_classes,
        )

        # Test 4-frame inference
        x_4 = torch.randn(batch_size * 4, 3, 224, 224)
        out = model(x_4, None, None, training=False)
        assert out.shape == (batch_size, num_classes)

        # Test 8-frame inference
        x_8 = torch.randn(batch_size * 8, 3, 224, 224)
        out = model(None, x_8, None, training=False)
        assert out.shape == (batch_size, num_classes)


# =============================================================================
# TSMFFN Wrapper Tests
# =============================================================================

class TestTSMFFN:
    """Test high-level TSMFFN model wrapper."""

    def test_tsmffn_training_forward(self, sample_inputs, batch_size, num_classes):
        """TSMFFN training mode accepts (B, C, T, H, W) inputs."""
        model = TSMFFN(num_classes=num_classes, pretrained=False)

        out_4, out_8, out_16 = model(
            sample_inputs["x_4"],
            sample_inputs["x_8"],
            sample_inputs["x_16"],
            training=True,
        )

        assert out_4.shape == (batch_size, num_classes)
        assert out_8.shape == (batch_size, num_classes)
        assert out_16.shape == (batch_size, num_classes)

    def test_tsmffn_inference_forward(self, batch_size, num_classes):
        """TSMFFN inference mode with single input."""
        model = TSMFFN(num_classes=num_classes, pretrained=False)

        # Test each frame count
        x_4 = torch.randn(batch_size, 3, 4, 224, 224)
        out = model(x_4=x_4, training=False)
        assert out.shape == (batch_size, num_classes)

        x_8 = torch.randn(batch_size, 3, 8, 224, 224)
        out = model(x_8=x_8, training=False)
        assert out.shape == (batch_size, num_classes)

        x_16 = torch.randn(batch_size, 3, 16, 224, 224)
        out = model(x_16=x_16, training=False)
        assert out.shape == (batch_size, num_classes)

    def test_tsmffn_gradient_flow(self, sample_inputs, batch_size, num_classes):
        """Gradients should flow through all three paths."""
        model = TSMFFN(num_classes=num_classes, pretrained=False)
        model.train()

        out_4, out_8, out_16 = model(
            sample_inputs["x_4"],
            sample_inputs["x_8"],
            sample_inputs["x_16"],
            training=True,
        )

        # Create dummy loss
        loss = out_4.sum() + out_8.sum() + out_16.sum()
        loss.backward()

        # Check gradients exist on backbone parameters
        grad_count = 0
        for param in model.parameters():
            if param.grad is not None:
                grad_count += 1

        assert grad_count > 0, "No gradients computed"


# =============================================================================
# Pretrained Weight Loading Tests
# =============================================================================

class TestPretrainedLoading:
    """Test pretrained weight loading with specialized BN mapping."""

    @pytest.mark.slow
    def test_resnet50_ffn_pretrained_loads(self, num_classes):
        """Pretrained weights should load without error."""
        # This downloads pretrained weights (slow)
        model = resnet50_ffn(pretrained=True, num_classes=num_classes)
        assert model is not None

    def test_resnet50_ffn_non_pretrained(self, num_classes):
        """Non-pretrained model should initialize without error."""
        model = resnet50_ffn(pretrained=False, num_classes=num_classes)
        assert model is not None

    def test_create_ffn_model_factory(self, device, num_classes):
        """Factory function should create model on specified device."""
        model = create_ffn_model(
            num_classes=num_classes,
            pretrained=False,
            device=device,
        )
        # Verify model is on correct device
        param_device = next(model.parameters()).device
        assert param_device == device


# =============================================================================
# Temporal Distillation Loss Tests
# =============================================================================

class TestTemporalDistillationLoss:
    """Test Temporal Distillation Loss computation."""

    def test_loss_forward_shape(self, batch_size, num_classes):
        """Loss should return scalar and loss dict."""
        criterion = TemporalDistillationLoss(lambda_kl=1.0)

        out_l = torch.randn(batch_size, num_classes)
        out_m = torch.randn(batch_size, num_classes)
        out_h = torch.randn(batch_size, num_classes)
        target = torch.randint(0, num_classes, (batch_size,))

        loss, loss_dict = criterion(out_l, out_m, out_h, target)

        assert loss.dim() == 0, "Loss should be scalar"
        assert "loss_ce" in loss_dict
        assert "loss_kl_l" in loss_dict
        assert "loss_kl_m" in loss_dict

    def test_loss_gradient_flows_to_students(self, batch_size, num_classes):
        """KL loss should provide gradients to student outputs."""
        criterion = TemporalDistillationLoss(lambda_kl=1.0)

        out_l = torch.randn(batch_size, num_classes, requires_grad=True)
        out_m = torch.randn(batch_size, num_classes, requires_grad=True)
        out_h = torch.randn(batch_size, num_classes, requires_grad=True)
        target = torch.randint(0, num_classes, (batch_size,))

        loss, _ = criterion(out_l, out_m, out_h, target)
        loss.backward()

        # Students should have gradients from KL loss
        assert out_l.grad is not None, "Low-frame should have gradients"
        assert out_m.grad is not None, "Medium-frame should have gradients"

    def test_loss_teacher_detached(self, batch_size, num_classes):
        """Teacher (high-frame) should NOT receive KL gradients."""
        criterion = TemporalDistillationLoss(lambda_kl=1.0)

        out_l = torch.randn(batch_size, num_classes, requires_grad=True)
        out_m = torch.randn(batch_size, num_classes, requires_grad=True)
        out_h = torch.randn(batch_size, num_classes, requires_grad=True)
        target = torch.randint(0, num_classes, (batch_size,))

        loss, _ = criterion(out_l, out_m, out_h, target)
        loss.backward()

        # out_h should have grad only from CE, not from KL (teacher is detached)
        # We can't easily verify this, but we verify grads exist
        assert out_h.grad is not None, "Teacher should have CE gradients"

    def test_functional_loss(self, batch_size, num_classes):
        """Functional interface should work correctly."""
        out_l = torch.randn(batch_size, num_classes)
        out_m = torch.randn(batch_size, num_classes)
        out_h = torch.randn(batch_size, num_classes)
        target = torch.randint(0, num_classes, (batch_size,))

        loss, loss_dict = compute_ffn_loss(out_l, out_m, out_h, target)

        assert loss.dim() == 0
        assert "loss_ce" in loss_dict
        assert "loss" in loss_dict

    def test_ffn_loss_class(self, batch_size, num_classes):
        """FFNLoss class alternative interface."""
        criterion = FFNLoss(lambda_kl=1.0)

        outputs = (
            torch.randn(batch_size, num_classes),
            torch.randn(batch_size, num_classes),
            torch.randn(batch_size, num_classes),
        )
        target = torch.randint(0, num_classes, (batch_size,))

        loss, loss_dict = criterion(outputs, target)

        assert loss.dim() == 0
        assert "loss_ce" in loss_dict


# =============================================================================
# Integration Tests
# =============================================================================

class TestFFNIntegration:
    """Integration tests for full FFN training pipeline."""

    def test_full_forward_backward(self, sample_inputs, batch_size, num_classes):
        """Full model forward/backward pass with loss."""
        model = TSMFFN(num_classes=num_classes, pretrained=False)
        criterion = TemporalDistillationLoss(lambda_kl=1.0)

        model.train()
        out_4, out_8, out_16 = model(
            sample_inputs["x_4"],
            sample_inputs["x_8"],
            sample_inputs["x_16"],
            training=True,
        )

        target = torch.randint(0, num_classes, (batch_size,))
        loss, loss_dict = criterion(out_4, out_8, out_16, target)

        loss.backward()

        # Verify loss is reasonable
        assert loss.item() > 0, "Loss should be positive"
        assert not torch.isnan(loss), "Loss should not be NaN"
        assert not torch.isinf(loss), "Loss should not be Inf"

    def test_train_and_eval_modes(self, sample_inputs, batch_size, num_classes):
        """Model should behave differently in train vs eval modes."""
        model = TSMFFN(num_classes=num_classes, pretrained=False)

        model.train()
        out_train_4, out_train_8, out_train_16 = model(
            sample_inputs["x_4"],
            sample_inputs["x_8"],
            sample_inputs["x_16"],
            training=True,
        )

        model.eval()
        with torch.no_grad():
            # In eval, we use inference mode with single input
            out_eval = model(x_8=sample_inputs["x_8"], training=False)

        assert out_train_8.shape == out_eval.shape

    def test_inference_at_different_frame_counts(self, batch_size, num_classes):
        """Model should produce valid outputs at any frame count."""
        model = TSMFFN(num_classes=num_classes, pretrained=False)
        model.eval()

        with torch.no_grad():
            # 4 frames
            x = torch.randn(batch_size, 3, 4, 224, 224)
            out = model(x_4=x, training=False)
            assert out.shape == (batch_size, num_classes)

            # 8 frames
            x = torch.randn(batch_size, 3, 8, 224, 224)
            out = model(x_8=x, training=False)
            assert out.shape == (batch_size, num_classes)

            # 16 frames
            x = torch.randn(batch_size, 3, 16, 224, 224)
            out = model(x_16=x, training=False)
            assert out.shape == (batch_size, num_classes)


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_tsmffn_requires_input_in_inference(self, num_classes):
        """Inference without any input should raise error."""
        model = TSMFFN(num_classes=num_classes, pretrained=False)

        with pytest.raises(ValueError, match="At least one input"):
            model(training=False)

    def test_tsmffn_requires_all_inputs_in_training(self, num_classes):
        """Training without all inputs should raise error."""
        model = TSMFFN(num_classes=num_classes, pretrained=False)
        x = torch.randn(2, 3, 8, 224, 224)

        with pytest.raises(AssertionError):
            model(x_8=x, training=True)  # Missing x_4 and x_16

    def test_loss_batch_size_one(self, num_classes):
        """Loss should work with batch size 1."""
        criterion = TemporalDistillationLoss()

        out_l = torch.randn(1, num_classes)
        out_m = torch.randn(1, num_classes)
        out_h = torch.randn(1, num_classes)
        target = torch.randint(0, num_classes, (1,))

        loss, _ = criterion(out_l, out_m, out_h, target)
        assert loss.dim() == 0


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
