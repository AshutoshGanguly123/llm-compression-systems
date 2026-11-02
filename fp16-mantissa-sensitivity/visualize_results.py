"""
Visualize experiment results: Plot how metrics vary with mantissa bits zeroed.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configuration
RESULTS_DIR = "results"
OUTPUT_DIR = "visualizations"

# Test names and their available metrics
TESTS = {
    "hellaswag": ["acc,none", "acc_norm,none"],
    "piqa": ["acc,none", "acc_norm,none"],
    "arc_easy": ["acc,none", "acc_norm,none"],
    "winogrande": ["acc,none"],
    "boolq": ["acc,none"],
}

# Test display names
TEST_DISPLAY_NAMES = {
    "hellaswag": "HellaSwag",
    "piqa": "PIQA",
    "arc_easy": "ARC Easy",
    "winogrande": "Winogrande",
    "boolq": "BoolQ",
}

# Metric display names
METRIC_DISPLAY_NAMES = {
    "acc,none": "Accuracy",
    "acc_norm,none": "Normalized Accuracy",
}


def load_all_experiments(results_dir):
    """Load all experiment JSON files."""
    experiments = {}
    
    for num_bits in range(10):  # 0-9 bits
        filepath = os.path.join(results_dir, f"experiment_{num_bits}_bits.json")
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
                experiments[num_bits] = data
        else:
            print(f"Warning: {filepath} not found")
    
    return experiments


def extract_metric_data(experiments, test_name, metric_name):
    """Extract metric values and errors across all experiments."""
    bits = []
    values = []
    errors = []
    
    for num_bits in sorted(experiments.keys()):
        exp = experiments[num_bits]
        if test_name in exp.get("results", {}):
            test_results = exp["results"][test_name]
            if metric_name in test_results:
                bits.append(num_bits)
                values.append(test_results[metric_name])
                # Get corresponding error if available
                error_key = metric_name.replace("acc", "acc_stderr")
                if error_key in test_results:
                    errors.append(test_results[error_key])
                else:
                    errors.append(0.0)
    
    return np.array(bits), np.array(values), np.array(errors)


def plot_test_metrics(experiments, test_name, output_dir):
    """Create plots for all metrics of a specific test."""
    metrics = TESTS[test_name]
    n_metrics = len(metrics)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]
    
    fig.suptitle(f"{TEST_DISPLAY_NAMES[test_name]} - Performance vs Mantissa Bits Zeroed", 
                 fontsize=14, fontweight='bold')
    
    for idx, metric_name in enumerate(metrics):
        ax = axes[idx]
        
        # Extract data
        bits, values, errors = extract_metric_data(experiments, test_name, metric_name)
        
        if len(bits) == 0:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            ax.set_xlabel("Bits Zeroed")
            continue
        
        # Plot line with error bars
        ax.errorbar(bits, values, yerr=errors, marker='o', linestyle='-', 
                   linewidth=2, markersize=8, capsize=5, capthick=2,
                   color='#2E86AB', ecolor='#A23B72', alpha=0.8)
        
        # Formatting
        ax.set_xlabel("Number of Mantissa Bits Zeroed", fontsize=11)
        ax.set_ylabel(f"{METRIC_DISPLAY_NAMES.get(metric_name, metric_name)}", fontsize=11)
        ax.set_title(f"{METRIC_DISPLAY_NAMES.get(metric_name, metric_name)}", fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xticks(bits)
        ax.set_xlim(-0.5, max(bits) + 0.5)
        
        # Add baseline reference line
        if 0 in bits:
            baseline_idx = np.where(bits == 0)[0][0]
            baseline_value = values[baseline_idx]
            ax.axhline(y=baseline_value, color='green', linestyle='--', 
                      alpha=0.5, linewidth=1, label='Baseline (0 bits)')
            ax.legend(loc='best')
        
        # Format y-axis to show percentages
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{test_name}_metrics.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def plot_all_tests_overview(experiments, output_dir):
    """Create a combined overview plot with all tests."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    fig.suptitle("All Tests - Accuracy vs Mantissa Bits Zeroed", 
                 fontsize=16, fontweight='bold')
    
    test_idx = 0
    for test_name in TESTS.keys():
        ax = axes[test_idx]
        
        # Use acc,none for all tests (it's available for all)
        metric_name = "acc,none"
        bits, values, errors = extract_metric_data(experiments, test_name, metric_name)
        
        if len(bits) > 0:
            ax.errorbar(bits, values, yerr=errors, marker='o', linestyle='-', 
                       linewidth=2, markersize=6, capsize=4, capthick=1.5,
                       color='#2E86AB', ecolor='#A23B72', alpha=0.8)
            
            # Add baseline reference
            if 0 in bits:
                baseline_idx = np.where(bits == 0)[0][0]
                baseline_value = values[baseline_idx]
                ax.axhline(y=baseline_value, color='green', linestyle='--', 
                          alpha=0.5, linewidth=1)
        
        ax.set_title(TEST_DISPLAY_NAMES[test_name], fontsize=12, fontweight='bold')
        ax.set_xlabel("Bits Zeroed", fontsize=10)
        ax.set_ylabel("Accuracy", fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xticks(range(0, 10))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        test_idx += 1
    
    # Hide the last subplot if we have an odd number
    if test_idx < len(axes):
        axes[test_idx].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, "all_tests_overview.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def plot_relative_degradation(experiments, output_dir):
    """Plot relative performance degradation from baseline."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    
    # Get baseline values (0 bits)
    baseline_values = {}
    if 0 in experiments:
        baseline = experiments[0]["results"]
        for test_name in TESTS.keys():
            if test_name in baseline and "acc,none" in baseline[test_name]:
                baseline_values[test_name] = baseline[test_name]["acc,none"]
    
    # Plot relative change for each test
    colors = plt.cm.tab10(np.linspace(0, 1, len(TESTS)))
    
    for idx, test_name in enumerate(TESTS.keys()):
        if test_name not in baseline_values:
            continue
        
        bits, values, errors = extract_metric_data(experiments, test_name, "acc,none")
        if len(bits) == 0:
            continue
        
        # Calculate relative change from baseline
        baseline_val = baseline_values[test_name]
        relative_change = ((values - baseline_val) / baseline_val) * 100
        
        ax.plot(bits, relative_change, marker='o', linestyle='-', 
               linewidth=2, markersize=8, label=TEST_DISPLAY_NAMES[test_name],
               color=colors[idx], alpha=0.8)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    ax.set_xlabel("Number of Mantissa Bits Zeroed", fontsize=12)
    ax.set_ylabel("Relative Change from Baseline (%)", fontsize=12)
    ax.set_title("Performance Degradation vs Mantissa Bits Zeroed", 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10)
    ax.set_xticks(range(0, 10))
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, "relative_degradation.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def main():
    """Main function to generate all visualizations."""
    print("=" * 80)
    print("Generating Visualizations from Experiment Results")
    print("=" * 80)
    print()
    
    # Load all experiments
    print("Loading experiment results...")
    experiments = load_all_experiments(RESULTS_DIR)
    print(f"Loaded {len(experiments)} experiments: {sorted(experiments.keys())}")
    print()
    
    if len(experiments) == 0:
        print("Error: No experiment files found!")
        return
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate individual test plots
    print("Generating individual test plots...")
    for test_name in TESTS.keys():
        plot_test_metrics(experiments, test_name, OUTPUT_DIR)
    print()
    
    # Generate overview plot
    print("Generating overview plot...")
    plot_all_tests_overview(experiments, OUTPUT_DIR)
    print()
    
    # Generate relative degradation plot
    print("Generating relative degradation plot...")
    plot_relative_degradation(experiments, OUTPUT_DIR)
    print()
    
    print("=" * 80)
    print(f"All visualizations saved to: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 80)


if __name__ == "__main__":
    main()

