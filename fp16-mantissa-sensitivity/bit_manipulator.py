"""
Bit manipulator for zeroing out LSBs in FP16 mantissa.
"""

import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def zero_mantissa_lsbs(weight_tensor, num_bits):
    """
    Zero out the least significant bits in the FP16 mantissa.
    
    FP16 format: 1 sign bit + 5 exponent bits + 10 mantissa bits
    Mantissa bits are bits 0-9 (LSB to MSB of mantissa)
    
    Args:
        weight_tensor: FP16 tensor to modify
        num_bits: Number of LSBs in mantissa to zero (1-9)
    
    Returns:
        Modified tensor with specified mantissa LSBs set to zero
    
    Raises:
        ValueError: If num_bits is out of range or tensor is not FP16
    """
    if num_bits < 1 or num_bits > 9:
        raise ValueError(f"num_bits must be between 1 and 9, got {num_bits}")
    
    if weight_tensor.dtype != torch.float16:
        raise ValueError(f"Expected FP16 tensor, got {weight_tensor.dtype}")
    
    # Convert FP16 to int32 for bit operations (CUDA supports int32 bitwise ops)
    # First view as uint16, then convert to int32 to avoid precision issues
    uint16_tensor = weight_tensor.view(torch.uint16)
    int32_tensor = uint16_tensor.to(torch.int32)  # Convert to int32 for CUDA compatibility
    
    # Create mask to zero the last num_bits of mantissa
    # FP16 format: bit 15 (sign), bits 14-10 (exponent), bits 9-0 (mantissa)
    # We want to zero bits 0 to (num_bits-1) in the mantissa
    # Example: num_bits=2 -> mask = 0xFFFC (preserves sign+exponent, zeros last 2 mantissa bits)
    # Create mask: preserve all bits except the last num_bits of mantissa
    full_mask = 0xFFFF & (~((1 << num_bits) - 1))
    
    # Apply mask: zero the specified LSBs in mantissa (preserves sign and exponent)
    # Use int32 for bitwise operations (CUDA compatible)
    masked_tensor = int32_tensor & full_mask
    
    # Convert back: int32 -> uint16 -> float16
    uint16_result = masked_tensor.to(torch.uint16)
    result = uint16_result.view(torch.float16)
    
    return result


def apply_to_model_weights(model, num_bits):
    """
    Apply mantissa LSB zeroing to all weight parameters in the model.
    
    Args:
        model: PyTorch model
        num_bits: Number of mantissa LSBs to zero (1-9)
    
    Returns:
        Number of parameters modified
    """
    modified_count = 0
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            # Only modify weight tensors (skip biases if they exist)
            # Typically, weights are 2D+ tensors, biases are 1D
            if param.requires_grad and param.dtype == torch.float16:
                # Check if it's a weight (not a bias)
                # Biases are typically 1D, but we'll modify all FP16 params
                # to be safe, you can add filtering here if needed
                try:
                    param.data = zero_mantissa_lsbs(param.data, num_bits)
                    modified_count += 1
                except ValueError as e:
                    logger.warning(f"Skipping parameter {name}: {e}")
    
    logger.info(f"Modified {modified_count} parameters")
    return modified_count


def verify_bit_zeroing(tensor, num_bits):
    """
    Verify that the specified number of LSBs in mantissa are zero.
    
    Args:
        tensor: FP16 tensor to verify
        num_bits: Expected number of zeroed LSBs
    
    Returns:
        bool: True if verification passes
    """
    if tensor.dtype != torch.float16:
        return False
    
    # Convert to int32 for CUDA compatibility (CUDA doesn't support uint16 bitwise ops)
    uint16_tensor = tensor.view(torch.uint16)
    int32_tensor = uint16_tensor.to(torch.int32)  # Convert to int32 for CUDA compatibility
    
    # Check that the last num_bits of mantissa are zero
    # Mantissa bits are bits 0-9
    mask = (1 << num_bits) - 1  # Bits that should be zero
    
    # Check a sample of values
    sample = int32_tensor.flatten()[:1000]  # Check first 1000 values
    
    # Extract mantissa bits (bits 0-9) using int32 bitwise operations
    mantissa_bits = sample & 0x3FF
    
    # Check that the last num_bits are zero
    masked_bits = mantissa_bits & mask
    
    # All masked bits should be zero
    all_zeroed = torch.all(masked_bits == 0)
    
    return all_zeroed.item() if isinstance(all_zeroed, torch.Tensor) else all_zeroed

