"""
Main experiment runner for FP16 mantissa bit sensitivity experiment.
"""

import os
import sys
import logging
import torch
from config import (
    MODEL_NAME,
    DEVICE,
    SELECTED_TESTS,
    OUTPUT_DIR,
    BASELINE_EXPERIMENT_ID,
    MAX_BITS_TO_ZERO,
)
from model_loader import load_model_fp16
from bit_manipulator import apply_to_model_weights
from test_runner import run_benchmarks
from results_manager import save_results, generate_summary_report

# Check for HuggingFace token
if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGINGFACE_HUB_TOKEN"):
    logging.warning("HF_TOKEN not set. If accessing gated models, set: export HF_TOKEN=your_token")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_completed_experiments(output_dir):
    """Get list of completed experiment bit counts."""
    completed = []
    for num_bits in range(0, MAX_BITS_TO_ZERO + 1):
        result_file = os.path.join(output_dir, f"experiment_{num_bits}_bits.json")
        if os.path.exists(result_file):
            completed.append(num_bits)
    return completed


def run_experiment():
    """
    Run the complete experiment: progressively zero mantissa bits and measure performance.
    """
    logger.info("=" * 80)
    logger.info("Starting FP16 Mantissa Bit Sensitivity Experiment")
    logger.info("=" * 80)
    logger.info(f"Model: {MODEL_NAME}")
    
    # Check GPU availability - fail if not available
    if DEVICE is None or not torch.cuda.is_available():
        logger.error("=" * 80)
        logger.error("ERROR: GPU not available!")
        logger.error("Please connect to a GPU node before running this experiment.")
        logger.error("Example: srun --partition=gpu --nodelist=gpu-01 --pty bash")
        logger.error("=" * 80)
        sys.exit(1)
    
    device = DEVICE
    gpu_id = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(gpu_id)
    
    logger.info(f"Device: {device}")
    logger.info(f"GPU ID: {gpu_id}")
    logger.info(f"GPU Name: {gpu_name}")
    logger.info(f"Tests: {SELECTED_TESTS}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info("")
    
    try:
        # Check which experiments are already completed
        completed = get_completed_experiments(OUTPUT_DIR)
        logger.info(f"Found {len(completed)} completed experiments: {sorted(completed)}")
        logger.info("")
        
        # Step 1: Load baseline model (only if baseline not done)
        if 0 not in completed:
            logger.info("Step 1: Loading baseline model in FP16...")
            model, tokenizer = load_model_fp16(MODEL_NAME, device=device)
            logger.info("Baseline model loaded successfully")
            logger.info("")
            
            # Step 2: Run baseline benchmarks
            logger.info("Step 2: Running baseline benchmarks (0 bits zeroed)...")
            baseline_results = run_benchmarks(
                model,
                tokenizer,
                SELECTED_TESTS,
                device=device,
            )
            
            # Save baseline results
            model_config = {
                "model_name": MODEL_NAME,
                "device": device,
                "dtype": "float16",
            }
            
            save_results(
                experiment_id=BASELINE_EXPERIMENT_ID,
                num_bits_zeroed=0,
                results=baseline_results,
                output_dir=OUTPUT_DIR,
                model_config=model_config,
            )
            logger.info("Baseline benchmarks completed and saved")
            logger.info("")
        else:
            logger.info("Step 1-2: Baseline already completed, skipping...")
            logger.info("")
            model_config = {
                "model_name": MODEL_NAME,
                "device": device,
                "dtype": "float16",
            }
        
        # Step 3: Progressive bit zeroing experiments
        logger.info("Step 3: Starting progressive mantissa bit zeroing experiments...")
        logger.info("")
        
        for num_bits in range(1, MAX_BITS_TO_ZERO + 1):
            # Skip if already completed
            if num_bits in completed:
                logger.info(f"Skipping experiment {num_bits} bits (already completed)")
                logger.info("")
                continue
            
            logger.info("-" * 80)
            logger.info(f"Experiment: Zeroing {num_bits} LSB(s) in mantissa")
            logger.info("-" * 80)
            
            try:
                # Reload model for each experiment (bit zeroing is irreversible)
                logger.info(f"Reloading model for experiment with {num_bits} bits...")
                experiment_model, experiment_tokenizer = load_model_fp16(MODEL_NAME, device=device)
                
                # Apply bit zeroing to model weights
                logger.info(f"Zeroing {num_bits} LSB(s) in all model weights...")
                modified_count = apply_to_model_weights(experiment_model, num_bits)
                logger.info(f"Modified {modified_count} parameters")
                
                # Run benchmarks
                logger.info(f"Running benchmarks with {num_bits} bit(s) zeroed...")
                experiment_results = run_benchmarks(
                    experiment_model,
                    experiment_tokenizer,
                    SELECTED_TESTS,
                    device=device,
                )
                
                # Save results
                experiment_id = f"experiment_{num_bits}_bits"
                save_results(
                    experiment_id=experiment_id,
                    num_bits_zeroed=num_bits,
                    results=experiment_results,
                    output_dir=OUTPUT_DIR,
                    model_config=model_config,
                )
                
                # Clean up model to free memory
                del experiment_model
                del experiment_tokenizer
                if device == "cuda":
                    torch.cuda.empty_cache()
                
                logger.info(f"Experiment {num_bits} bits completed and saved")
                logger.info("")
                
            except Exception as e:
                logger.error(f"Failed in experiment with {num_bits} bits zeroed: {e}")
                logger.error("Continuing with next experiment...")
                logger.info("")
                continue
        
        # Step 4: Generate summary report
        logger.info("=" * 80)
        logger.info("Step 4: Generating summary report...")
        logger.info("=" * 80)
        
        summary = generate_summary_report(OUTPUT_DIR)
        
        if summary:
            logger.info("Summary report generated successfully")
            logger.info("")
            logger.info("Experiment Summary:")
            logger.info(f"  Baseline: 0 bits zeroed")
            for exp in summary.get("experiments", []):
                bits = exp["num_bits_zeroed"]
                logger.info(f"  {bits} bits zeroed: Results saved")
        else:
            logger.warning("Could not generate summary report")
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("Experiment completed!")
        logger.info(f"Results saved in: {os.path.abspath(OUTPUT_DIR)}")
        logger.info("=" * 80)
        
    except KeyboardInterrupt:
        logger.warning("Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Experiment failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    run_experiment()

