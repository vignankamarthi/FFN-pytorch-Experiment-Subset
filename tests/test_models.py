"""Test script for Phase 4: TSM Model Architecture.

Run with: python -m tests.test_model (from project root)
      or: pytest tests/test_model.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.models import TSMResNet50, TemporalShift, temporal_shift


def test_temporal_shift_function():
    """Test the temporal_shift function."""
    print("\n=== Test: temporal_shift function ===")

    # Create dummy input: B=2, T=8, C=64, H=56, W=56
    b, t, c, h, w = 2, 8, 64, 56, 56
    x = torch.randn(b * t, c, h, w)

    # Apply shift
    out = temporal_shift(x, num_frames=t, shift_div=8)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")

    assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"

    # Check that shifting actually happened
    # First 1/8 channels should be shifted forward
    fold = c // 8
    x_reshaped = x.view(b, t, c, h, w)
    out_reshaped = out.view(b, t, c, h, w)

    # Frame 0's first fold channels should be from frame 1
    # (except last frame which gets zeros)
    assert torch.allclose(
        out_reshaped[:, :-1, :fold],
        x_reshaped[:, 1:, :fold]
    ), "Forward shift not working correctly"

    print("temporal_shift function: PASSED")


def test_temporal_shift_module():
    """Test the TemporalShift nn.Module wrapper."""
    print("\n=== Test: TemporalShift module ===")

    # Create a dummy conv layer to wrap
    conv = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
    tsm = TemporalShift(conv, num_frames=8, shift_div=8)

    # Test forward
    x = torch.randn(16, 64, 56, 56)  # B=2, T=8
    out = tsm(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")

    assert out.shape == (16, 128, 56, 56), f"Expected (16, 128, 56, 56), got {out.shape}"

    print("TemporalShift module: PASSED")


def test_tsm_resnet50_forward():
    """Test TSMResNet50 forward pass."""
    print("\n=== Test: TSMResNet50 forward pass ===")

    # Create model (use CPU for testing)
    model = TSMResNet50(
        num_classes=174,
        num_frames=16,
        pretrained=True,
    )
    model.eval()

    # Test input: B=2, C=3, T=16, H=224, W=224
    x = torch.randn(2, 3, 16, 224, 224)

    with torch.no_grad():
        out = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")

    assert out.shape == (2, 174), f"Expected (2, 174), got {out.shape}"

    print("TSMResNet50 forward: PASSED")


def test_tsm_resnet50_backward():
    """Test TSMResNet50 backward pass (gradient flow)."""
    print("\n=== Test: TSMResNet50 backward pass ===")

    model = TSMResNet50(
        num_classes=174,
        num_frames=16,
        pretrained=True,
    )
    model.train()

    # Smaller input for faster test
    x = torch.randn(1, 3, 8, 224, 224, requires_grad=True)

    # Need to recreate model with correct num_frames
    model = TSMResNet50(
        num_classes=174,
        num_frames=8,
        pretrained=True,
    )
    model.train()

    out = model(x)
    loss = out.sum()
    loss.backward()

    # Check that gradients exist
    assert x.grad is not None, "No gradient for input"

    # Check model parameters have gradients
    grad_count = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_count += 1

    print(f"Parameters with gradients: {grad_count}")
    assert grad_count > 0, "No parameters have gradients"

    print("TSMResNet50 backward: PASSED")


def test_tsm_resnet50_multi_frame():
    """Test TSMResNet50 with different frame counts."""
    print("\n=== Test: TSMResNet50 multi-frame ===")

    for num_frames in [4, 8, 16]:
        model = TSMResNet50(
            num_classes=174,
            num_frames=num_frames,
            pretrained=True,
        )
        model.eval()

        x = torch.randn(1, 3, num_frames, 224, 224)

        with torch.no_grad():
            out = model(x)

        print(f"  T={num_frames}: input {x.shape} -> output {out.shape}")
        assert out.shape == (1, 174), f"Expected (1, 174), got {out.shape}"

    print("TSMResNet50 multi-frame: PASSED")


def test_parameter_count():
    """Verify model has approximately 25.6M parameters."""
    print("\n=== Test: Parameter count ===")

    model = TSMResNet50(
        num_classes=174,
        num_frames=16,
        pretrained=True,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total in millions: {total_params / 1e6:.2f}M")

    # ResNet-50 has ~25.6M params, our FC layer changes it slightly
    # (1000 -> 174 classes means fewer output params)
    # Expected: around 23-26M
    assert 20_000_000 < total_params < 30_000_000, \
        f"Parameter count {total_params:,} outside expected range (20M-30M)"

    print("Parameter count: PASSED")


if __name__ == "__main__":
    print("=" * 50)
    print("Phase 4: TSM Model Architecture Tests")
    print("=" * 50)

    test_temporal_shift_function()
    test_temporal_shift_module()
    test_tsm_resnet50_forward()
    test_tsm_resnet50_backward()
    test_tsm_resnet50_multi_frame()
    test_parameter_count()

    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)
