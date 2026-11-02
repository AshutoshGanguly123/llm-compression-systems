"""
Configuration constants for the FP16 mantissa bit sensitivity experiment.
"""

import torch

# Model configuration
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"  # Llama 3.1 8B Instruct model

# GPU configuration - always use CUDA if available
# Will fail if GPU not available (no CPU fallback)
if torch.cuda.is_available():
    DEVICE = "cuda"
    GPU_ID = torch.cuda.current_device() if torch.cuda.is_available() else 0
    GPU_NAME = torch.cuda.get_device_name(GPU_ID)
else:
    DEVICE = None  # Will cause error if GPU not available
    GPU_ID = None
    GPU_NAME = None

# Selected lightweight tests from lm-evaluation-harness
SELECTED_TESTS = [
    "hellaswag",
    "piqa",
    "arc_easy",
    "winogrande",
    "boolq"
]

# Experiment configuration
OUTPUT_DIR = "results"
BASELINE_EXPERIMENT_ID = "experiment_0_bits"
NUM_MANTISSA_BITS = 10  # FP16 has 10 mantissa bits
MAX_BITS_TO_ZERO = 9    # Zero 1-9 bits, leaving at least 1 bit

# Results file naming
RESULTS_FILE_PATTERN = "experiment_{num_bits}_bits.json"
SUMMARY_REPORT_FILE = "summary_report.json"

