"""
Results manager for saving, loading, and comparing experiment results.
"""

import json
import os
from datetime import datetime
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_results(experiment_id, num_bits_zeroed, results, output_dir, model_config=None):
    """
    Save experiment results to a JSON file.
    
    Args:
        experiment_id: Unique identifier for the experiment
        num_bits_zeroed: Number of mantissa bits that were zeroed
        results: Dictionary of benchmark results
        output_dir: Directory to save results in
        model_config: Optional model configuration dict
    
    Returns:
        str: Path to saved file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare result data
    result_data = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now().isoformat(),
        "num_bits_zeroed": num_bits_zeroed,
        "model_config": model_config or {},
        "results": results,
    }
    
    # Generate filename
    filename = f"experiment_{num_bits_zeroed}_bits.json"
    filepath = os.path.join(output_dir, filename)
    
    # Save to JSON
    with open(filepath, "w") as f:
        json.dump(result_data, f, indent=2)
    
    logger.info(f"Saved results to {filepath}")
    return filepath


def load_results(output_dir, num_bits_zeroed=None):
    """
    Load experiment results from JSON file(s).
    
    Args:
        output_dir: Directory containing results
        num_bits_zeroed: Optional specific number of bits to load
    
    Returns:
        dict or list: Single result dict if num_bits_zeroed specified, else list of all results
    """
    output_path = Path(output_dir)
    
    if not output_path.exists():
        logger.warning(f"Output directory does not exist: {output_dir}")
        return [] if num_bits_zeroed is None else None
    
    if num_bits_zeroed is not None:
        # Load specific file
        filename = f"experiment_{num_bits_zeroed}_bits.json"
        filepath = output_path / filename
        
        if not filepath.exists():
            logger.warning(f"Results file does not exist: {filepath}")
            return None
        
        with open(filepath, "r") as f:
            return json.load(f)
    else:
        # Load all result files
        results = []
        for filepath in sorted(output_path.glob("experiment_*_bits.json")):
            try:
                with open(filepath, "r") as f:
                    results.append(json.load(f))
            except Exception as e:
                logger.warning(f"Failed to load {filepath}: {e}")
        
        return results


def compare_results(baseline, modified):
    """
    Compare baseline and modified results to calculate performance drops.
    
    Args:
        baseline: Baseline results dict (from load_results)
        modified: Modified results dict (from load_results)
    
    Returns:
        dict: Comparison metrics showing performance changes
    """
    comparison = {
        "baseline_bits_zeroed": baseline.get("num_bits_zeroed", 0),
        "modified_bits_zeroed": modified.get("num_bits_zeroed", 0),
        "tests": {},
    }
    
    baseline_results = baseline.get("results", {})
    modified_results = modified.get("results", {})
    
    # Compare each test
    all_tests = set(baseline_results.keys()) | set(modified_results.keys())
    
    for test_name in all_tests:
        baseline_test = baseline_results.get(test_name, {})
        modified_test = modified_results.get(test_name, {})
        
        test_comparison = {}
        
        # Compare all metrics
        all_metrics = set(baseline_test.keys()) | set(modified_test.keys())
        
        for metric_name in all_metrics:
            baseline_value = baseline_test.get(metric_name)
            modified_value = modified_test.get(metric_name)
            
            if baseline_value is not None and modified_value is not None:
                try:
                    baseline_float = float(baseline_value)
                    modified_float = float(modified_value)
                    
                    absolute_change = modified_float - baseline_float
                    relative_change = (absolute_change / baseline_float * 100) if baseline_float != 0 else 0
                    
                    test_comparison[metric_name] = {
                        "baseline": baseline_float,
                        "modified": modified_float,
                        "absolute_change": absolute_change,
                        "relative_change_percent": relative_change,
                    }
                except (ValueError, TypeError):
                    # Non-numeric metric, just store both values
                    test_comparison[metric_name] = {
                        "baseline": baseline_value,
                        "modified": modified_value,
                    }
        
        comparison["tests"][test_name] = test_comparison
    
    return comparison


def generate_summary_report(output_dir):
    """
    Generate a summary report comparing all experiments to baseline.
    
    Args:
        output_dir: Directory containing all result files
    
    Returns:
        dict: Summary report with all comparisons
    """
    # Load all results
    all_results = load_results(output_dir)
    
    if not all_results:
        logger.warning("No results found to generate summary")
        return {}
    
    # Find baseline (0 bits zeroed)
    baseline = None
    for result in all_results:
        if result.get("num_bits_zeroed", 0) == 0:
            baseline = result
            break
    
    if baseline is None:
        logger.warning("No baseline (0 bits) found")
        return {}
    
    # Compare each experiment to baseline
    summary = {
        "baseline": baseline,
        "experiments": [],
        "timestamp": datetime.now().isoformat(),
    }
    
    # Sort by num_bits_zeroed
    sorted_results = sorted(
        [r for r in all_results if r.get("num_bits_zeroed", 0) > 0],
        key=lambda x: x.get("num_bits_zeroed", 0)
    )
    
    for result in sorted_results:
        comparison = compare_results(baseline, result)
        summary["experiments"].append({
            "num_bits_zeroed": result.get("num_bits_zeroed"),
            "comparison": comparison,
        })
    
    # Save summary
    summary_path = os.path.join(output_dir, "summary_report.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Generated summary report: {summary_path}")
    return summary

