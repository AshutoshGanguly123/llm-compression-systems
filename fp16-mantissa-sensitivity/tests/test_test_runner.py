"""
Unit tests for test_runner.py
"""

import pytest
import torch
from model_loader import load_model_fp16
from test_runner import run_single_test, run_benchmarks


@pytest.fixture
def small_model():
    """Load a small model for testing."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model_fp16("gpt2", device=device)
    return model, tokenizer, device


def test_run_single_test(small_model):
    """Test running a single benchmark test."""
    model, tokenizer, device = small_model
    
    # Run a lightweight test with small limit
    results = run_single_test(
        model, 
        tokenizer, 
        "hellaswag", 
        device=device, 
        limit=5  # Very small limit for quick test
    )
    
    # Verify results structure
    assert isinstance(results, dict)
    assert "hellaswag" in results
    assert isinstance(results["hellaswag"], dict)


def test_run_benchmarks_multiple(small_model):
    """Test running multiple benchmarks."""
    model, tokenizer, device = small_model
    
    # Run multiple tests with small limits
    test_names = ["hellaswag", "piqa"]
    results = run_benchmarks(
        model,
        tokenizer,
        test_names,
        device=device,
        limit=5  # Small limit for quick test
    )
    
    # Verify results structure
    assert isinstance(results, dict)
    for test_name in test_names:
        assert test_name in results
        assert isinstance(results[test_name], dict)


def test_run_benchmarks_empty_list(small_model):
    """Test that empty test list raises error."""
    model, tokenizer, device = small_model
    
    with pytest.raises((RuntimeError, ValueError)):
        run_benchmarks(model, tokenizer, [], device=device, limit=5)


def test_run_benchmarks_invalid_test(small_model):
    """Test that invalid test name is handled gracefully."""
    model, tokenizer, device = small_model
    
    # This might fail or return empty results, but shouldn't crash
    try:
        results = run_benchmarks(
            model,
            tokenizer,
            ["nonexistent_test_12345"],
            device=device,
            limit=5
        )
        # If it doesn't raise, results should be empty or have error info
        assert isinstance(results, dict)
    except (RuntimeError, ValueError):
        # Expected behavior - invalid test should raise error
        pass

