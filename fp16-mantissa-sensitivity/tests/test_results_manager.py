"""
Unit tests for results_manager.py
"""

import pytest
import json
import os
import tempfile
import shutil
from results_manager import (
    save_results,
    load_results,
    compare_results,
    generate_summary_report
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_results():
    """Sample benchmark results for testing."""
    return {
        "hellaswag": {
            "acc": 0.75,
            "acc_norm": 0.76,
        },
        "piqa": {
            "acc": 0.70,
            "acc_norm": 0.71,
        },
    }


def test_save_results(temp_dir, sample_results):
    """Test saving results to file."""
    filepath = save_results(
        experiment_id="test_exp_1",
        num_bits_zeroed=2,
        results=sample_results,
        output_dir=temp_dir,
        model_config={"model_name": "test_model"}
    )
    
    # Verify file exists
    assert os.path.exists(filepath)
    assert filepath.endswith("experiment_2_bits.json")
    
    # Verify file contents
    with open(filepath, "r") as f:
        data = json.load(f)
    
    assert data["experiment_id"] == "test_exp_1"
    assert data["num_bits_zeroed"] == 2
    assert data["results"] == sample_results
    assert data["model_config"]["model_name"] == "test_model"
    assert "timestamp" in data


def test_load_results_specific(temp_dir, sample_results):
    """Test loading specific results file."""
    # Save a result
    save_results(
        experiment_id="test_exp",
        num_bits_zeroed=3,
        results=sample_results,
        output_dir=temp_dir,
    )
    
    # Load it
    loaded = load_results(temp_dir, num_bits_zeroed=3)
    
    assert loaded is not None
    assert loaded["num_bits_zeroed"] == 3
    assert loaded["results"] == sample_results


def test_load_results_all(temp_dir, sample_results):
    """Test loading all results."""
    # Save multiple results
    for bits in [0, 1, 2]:
        save_results(
            experiment_id=f"test_exp_{bits}",
            num_bits_zeroed=bits,
            results=sample_results,
            output_dir=temp_dir,
        )
    
    # Load all
    all_results = load_results(temp_dir)
    
    assert len(all_results) == 3
    assert all(r["num_bits_zeroed"] in [0, 1, 2] for r in all_results)


def test_load_results_nonexistent(temp_dir):
    """Test loading nonexistent results."""
    result = load_results(temp_dir, num_bits_zeroed=999)
    assert result is None


def test_compare_results(sample_results):
    """Test comparing baseline and modified results."""
    baseline = {
        "num_bits_zeroed": 0,
        "results": sample_results,
    }
    
    modified = {
        "num_bits_zeroed": 2,
        "results": {
            "hellaswag": {
                "acc": 0.70,  # Lower than baseline 0.75
                "acc_norm": 0.77,  # Higher than baseline 0.76
            },
            "piqa": {
                "acc": 0.65,  # Lower than baseline 0.70
                "acc_norm": 0.68,  # Lower than baseline 0.71
            },
        },
    }
    
    comparison = compare_results(baseline, modified)
    
    assert comparison["baseline_bits_zeroed"] == 0
    assert comparison["modified_bits_zeroed"] == 2
    assert "hellaswag" in comparison["tests"]
    assert "piqa" in comparison["tests"]
    
    # Check hellaswag acc comparison
    hellaswag_acc = comparison["tests"]["hellaswag"]["acc"]
    assert hellaswag_acc["baseline"] == 0.75
    assert hellaswag_acc["modified"] == 0.70
    assert hellaswag_acc["absolute_change"] == -0.05
    assert abs(hellaswag_acc["relative_change_percent"] - (-5.0/0.75*100)) < 0.1


def test_generate_summary_report(temp_dir, sample_results):
    """Test generating summary report."""
    # Save baseline and multiple experiments
    for bits in [0, 1, 2, 3]:
        save_results(
            experiment_id=f"test_exp_{bits}",
            num_bits_zeroed=bits,
            results=sample_results,
            output_dir=temp_dir,
        )
    
    # Generate summary
    summary = generate_summary_report(temp_dir)
    
    assert "baseline" in summary
    assert "experiments" in summary
    assert len(summary["experiments"]) == 3  # 1, 2, 3 bits (excluding baseline)
    
    # Verify summary file was created
    summary_path = os.path.join(temp_dir, "summary_report.json")
    assert os.path.exists(summary_path)
    
    # Verify experiments are sorted by num_bits_zeroed
    bits_zeroed = [exp["num_bits_zeroed"] for exp in summary["experiments"]]
    assert bits_zeroed == [1, 2, 3]


def test_generate_summary_no_baseline(temp_dir, sample_results):
    """Test summary generation when no baseline exists."""
    # Save only non-baseline results
    save_results(
        experiment_id="test_exp",
        num_bits_zeroed=1,
        results=sample_results,
        output_dir=temp_dir,
    )
    
    summary = generate_summary_report(temp_dir)
    assert summary == {}

