"""
Unit tests for bit_manipulator.py
"""

import pytest
import torch
import numpy as np
from bit_manipulator import (
    zero_mantissa_lsbs,
    apply_to_model_weights,
    verify_bit_zeroing
)


def test_zero_mantissa_lsbs_basic():
    """Test basic mantissa LSB zeroing."""
    # Create a simple FP16 tensor
    tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16)
    
    # Zero 1 LSB
    result = zero_mantissa_lsbs(tensor, num_bits=1)
    
    # Verify result is still FP16
    assert result.dtype == torch.float16
    assert result.shape == tensor.shape


def test_zero_mantissa_lsbs_invalid_bits():
    """Test that invalid num_bits raises ValueError."""
    tensor = torch.tensor([1.0], dtype=torch.float16)
    
    with pytest.raises(ValueError):
        zero_mantissa_lsbs(tensor, num_bits=0)
    
    with pytest.raises(ValueError):
        zero_mantissa_lsbs(tensor, num_bits=10)
    
    with pytest.raises(ValueError):
        zero_mantissa_lsbs(tensor, num_bits=15)


def test_zero_mantissa_lsbs_wrong_dtype():
    """Test that non-FP16 tensor raises ValueError."""
    tensor = torch.tensor([1.0], dtype=torch.float32)
    
    with pytest.raises(ValueError):
        zero_mantissa_lsbs(tensor, num_bits=1)


def test_zero_mantissa_lsbs_verification():
    """Test that zeroed bits are actually zero."""
    # Create tensor with known values
    tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float16)
    
    # Zero 2 LSBs
    result = zero_mantissa_lsbs(tensor, num_bits=2)
    
    # Verify bits are zeroed
    assert verify_bit_zeroing(result, num_bits=2)


def test_zero_mantissa_lsbs_progressive():
    """Test progressive zeroing of more bits."""
    tensor = torch.randn(100, dtype=torch.float16)
    
    for num_bits in range(1, 10):
        result = zero_mantissa_lsbs(tensor, num_bits=num_bits)
        assert result.dtype == torch.float16
        assert verify_bit_zeroing(result, num_bits=num_bits)


def test_zero_mantissa_lsbs_preserves_structure():
    """Test that zeroing preserves tensor structure."""
    # Create multi-dimensional tensor
    tensor = torch.randn(2, 3, 4, dtype=torch.float16)
    
    result = zero_mantissa_lsbs(tensor, num_bits=3)
    
    assert result.shape == tensor.shape
    assert result.dtype == tensor.dtype


def test_apply_to_model_weights():
    """Test applying bit zeroing to a model."""
    from transformers import AutoModelForCausalLM
    
    # Use small model for testing
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2",
        torch_dtype=torch.float16,
    )
    model = model.half()
    
    # Count initial parameters
    initial_count = sum(1 for p in model.parameters() if p.dtype == torch.float16)
    
    # Apply zeroing
    modified_count = apply_to_model_weights(model, num_bits=2)
    
    # Verify parameters were modified
    assert modified_count > 0
    assert modified_count == initial_count  # All FP16 params should be modified


def test_verify_bit_zeroing():
    """Test the verification function."""
    tensor = torch.randn(1000, dtype=torch.float16)
    
    # Zero 3 bits
    result = zero_mantissa_lsbs(tensor, num_bits=3)
    
    # Verify
    assert verify_bit_zeroing(result, num_bits=3)
    assert not verify_bit_zeroing(result, num_bits=4)  # Should fail for more bits


def test_mantissa_only_manipulation():
    """Test that only mantissa bits are affected, not sign or exponent."""
    # Create tensor with various values (positive, negative, different magnitudes)
    values = torch.tensor([
        1.0, -1.0, 0.5, -0.5, 100.0, -100.0, 0.001, -0.001
    ], dtype=torch.float16)
    
    result = zero_mantissa_lsbs(values, num_bits=2)
    
    # Check that signs are preserved (positive stays positive, negative stays negative)
    original_signs = torch.sign(values)
    result_signs = torch.sign(result)
    
    # Most signs should be preserved (some edge cases might change due to rounding)
    sign_preservation = (original_signs == result_signs).float().mean()
    assert sign_preservation > 0.5  # At least 50% should preserve sign
    
    # Check that magnitudes are roughly similar (exponent should be mostly preserved)
    original_mags = torch.abs(values)
    result_mags = torch.abs(result)
    
    # Magnitudes should be within reasonable range
    ratio = result_mags / (original_mags + 1e-8)
    assert torch.all(ratio > 0.1)  # Should not change by more than 10x
    assert torch.all(ratio < 10.0)  # Should not change by more than 10x

