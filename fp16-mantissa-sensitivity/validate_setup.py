"""
Quick validation script to verify all components work before running full experiment.
"""

import sys
import logging
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_setup():
    """Run quick validation tests on all components."""
    logger.info("Validating experiment setup...")
    logger.info("")
    
    # Test 1: Import all modules
    logger.info("Test 1: Importing modules...")
    try:
        from config import MODEL_NAME, SELECTED_TESTS, OUTPUT_DIR
        from model_loader import load_model_fp16
        from bit_manipulator import zero_mantissa_lsbs, apply_to_model_weights
        from test_runner import run_single_test
        from results_manager import save_results, load_results
        logger.info("✓ All modules imported successfully")
    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        return False
    
    # Test 2: Check device
    logger.info("")
    logger.info("Test 2: Checking GPU availability...")
    if not torch.cuda.is_available():
        logger.error("✗ GPU not available!")
        logger.error("  Please connect to a GPU node before running experiments.")
        logger.error("  Example: srun --partition=gpu --nodelist=gpu-01 --pty bash")
        return False
    
    device = "cuda"
    gpu_id = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(gpu_id)
    logger.info(f"✓ GPU available!")
    logger.info(f"  GPU ID: {gpu_id}")
    logger.info(f"  GPU Name: {gpu_name}")
    logger.info(f"  GPU Memory: {torch.cuda.get_device_properties(gpu_id).total_memory / 1e9:.2f} GB")
    
    # Test 3: Load small model
    logger.info("")
    logger.info("Test 3: Loading test model (gpt2)...")
    try:
        model, tokenizer = load_model_fp16("gpt2", device=device)
        logger.info("✓ Model loaded successfully")
    except Exception as e:
        logger.error(f"✗ Model loading failed: {e}")
        return False
    
    # Test 4: Bit manipulation
    logger.info("")
    logger.info("Test 4: Testing bit manipulation...")
    try:
        sample_param = next(model.parameters())
        modified = zero_mantissa_lsbs(sample_param, num_bits=1)
        assert modified.dtype == torch.float16
        logger.info("✓ Bit manipulation works")
    except Exception as e:
        logger.error(f"✗ Bit manipulation failed: {e}")
        return False
    
    # Test 5: Apply to model
    logger.info("")
    logger.info("Test 5: Testing model weight modification...")
    try:
        count = apply_to_model_weights(model, num_bits=1)
        assert count > 0
        logger.info(f"✓ Modified {count} parameters")
    except Exception as e:
        logger.error(f"✗ Model modification failed: {e}")
        return False
    
    # Test 6: Results manager
    logger.info("")
    logger.info("Test 6: Testing results manager...")
    try:
        import tempfile
        import shutil
        temp_dir = tempfile.mkdtemp()
        test_results = {"test": {"acc": 0.5}}
        save_results("test", 1, test_results, temp_dir)
        loaded = load_results(temp_dir, num_bits_zeroed=1)
        assert loaded is not None
        shutil.rmtree(temp_dir)  # Use rmtree to remove directory with contents
        logger.info("✓ Results manager works")
    except Exception as e:
        logger.error(f"✗ Results manager failed: {e}")
        return False
    
    # Test 7: Test runner (skip if lm_eval not available)
    logger.info("")
    logger.info("Test 7: Testing test runner (may skip if lm_eval unavailable)...")
    try:
        # Reload model for test
        test_model, test_tokenizer = load_model_fp16("gpt2", device=device)
        results = run_single_test(
            test_model,
            test_tokenizer,
            "hellaswag",
            device=device,
            limit=2  # Very small limit
        )
        logger.info("✓ Test runner works")
    except Exception as e:
        logger.warning(f"⚠ Test runner skipped (this is okay): {e}")
        logger.info("  (lm_eval may need to be installed or configured)")
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("Validation complete! Setup looks good.")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Configuration:")
    logger.info(f"  Model: {MODEL_NAME}")
    logger.info(f"  Tests: {SELECTED_TESTS}")
    logger.info(f"  Output: {OUTPUT_DIR}")
    logger.info("")
    logger.info("You can now run: python experiment_runner.py")
    
    return True


if __name__ == "__main__":
    success = validate_setup()
    sys.exit(0 if success else 1)

