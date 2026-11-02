"""
Unit tests for model_loader.py
"""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from model_loader import load_model_fp16, verify_model_fp16


@pytest.fixture
def small_model_name():
    """Use a small model for testing."""
    return "gpt2"  # Small model for quick tests


def test_load_model_fp16(small_model_name):
    """Test that model loads correctly in FP16."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model, tokenizer = load_model_fp16(small_model_name, device=device)
    
    # Verify model and tokenizer are returned
    assert model is not None
    assert tokenizer is not None
    
    # Verify model is in FP16
    assert verify_model_fp16(model)
    
    # Verify model is on correct device
    sample_param = next(model.parameters())
    assert sample_param.device.type == device


def test_model_forward_pass(small_model_name):
    """Test that model can perform forward pass."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model, tokenizer = load_model_fp16(small_model_name, device=device)
    model.eval()
    
    # Test forward pass
    test_text = "Hello, how are you?"
    inputs = tokenizer(test_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Verify output shape
    assert outputs.logits is not None
    assert outputs.logits.shape[0] == 1  # Batch size
    assert outputs.logits.shape[1] == inputs.input_ids.shape[1]  # Sequence length


def test_model_dtype_consistency(small_model_name):
    """Test that all parameters maintain FP16 dtype."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model, _ = load_model_fp16(small_model_name, device=device)
    
    # Check all parameters are FP16
    for name, param in model.named_parameters():
        assert param.dtype == torch.float16, f"Parameter {name} is not FP16: {param.dtype}"


def test_tokenizer_functionality(small_model_name):
    """Test that tokenizer works correctly."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    _, tokenizer = load_model_fp16(small_model_name, device=device)
    
    # Test encoding
    text = "Test sentence"
    encoded = tokenizer(text, return_tensors="pt")
    
    assert "input_ids" in encoded
    assert encoded["input_ids"].shape[0] == 1
    
    # Test decoding
    decoded = tokenizer.decode(encoded["input_ids"][0])
    assert isinstance(decoded, str)
    assert len(decoded) > 0

