"""
Test runner for executing lm-evaluation-harness benchmarks.
"""

import os
import torch
import logging
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomHFLM(HFLM):
    """
    Custom HuggingFace model wrapper that uses an already-loaded model.
    Override the internal methods that create model/tokenizer to return ours.
    """
    def __init__(self, model, tokenizer, device="cuda"):
        # Store our pre-loaded instances
        self._our_model = model
        self._our_tokenizer = tokenizer
        
        # Set model to eval mode
        model.eval()
        
        # CRITICAL: Set _model and _tokenizer BEFORE parent init
        # Parent accesses self.model (which returns self._model) at line 242 before calling _create_model
        self._model = model
        self._tokenizer = tokenizer
        
        # Initialize parent - it will call _create_model and _create_tokenizer
        # which we override to return our instances, but parent should see _model/_tokenizer already set
        super().__init__(
            pretrained=model.config.name_or_path if hasattr(model, 'config') else "gpt2",
            device=device,
        )
        
        # Ensure our instances are still set after parent init
        # (parent might have overwritten them)
        self._model = model
        self._tokenizer = tokenizer
    
    def _create_model(self, *args, **kwargs):
        """Override to return our pre-loaded model instead of creating new one."""
        return self._our_model
    
    def _create_tokenizer(self, *args, **kwargs):
        """Override to return our pre-loaded tokenizer instead of creating new one."""
        return self._our_tokenizer
    
    @property
    def tokenizer(self):
        """Override tokenizer property to always return our tokenizer."""
        # #region agent log
        with open(os.devnull, 'a') as f:
            import json, time
            has_tokenizer_attr = hasattr(self, '_tokenizer')
            tokenizer_attr = getattr(self, '_tokenizer', None)
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"I","location":"test_runner.py:110","message":"tokenizer property accessed","data":{"has_tokenizer_attr":has_tokenizer_attr,"tokenizer_attr_is_none":tokenizer_attr is None,"tokenizer_attr_type":type(tokenizer_attr).__name__ if tokenizer_attr else None,"has_our_tokenizer":hasattr(self,'_our_tokenizer'),"our_tokenizer_is_none":getattr(self,'_our_tokenizer',None) is None},"timestamp":int(time.time()*1000)}) + '\n')
        # #endregion
        # Always return our tokenizer, but also ensure _tokenizer is set for parent compatibility
        if not hasattr(self, '_tokenizer') or self._tokenizer is None:
            self._tokenizer = self._our_tokenizer
        return self._our_tokenizer
    
    @tokenizer.setter
    def tokenizer(self, value):
        """Override tokenizer setter to prevent parent from setting it to None."""
        # #region agent log
        with open(os.devnull, 'a') as f:
            import json, time
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"J","location":"test_runner.py:122","message":"tokenizer setter called","data":{"value_is_none":value is None,"value_type":type(value).__name__ if value else None},"timestamp":int(time.time()*1000)}) + '\n')
        # #endregion
        # Only set _tokenizer if value is not None, otherwise keep our tokenizer
        if value is not None:
            self._tokenizer = value
        # Always keep _our_tokenizer as backup


def run_benchmarks(model, tokenizer, test_names, device="cuda", limit=None):
    """
    Run lm-evaluation-harness benchmarks on the model.
    
    Args:
        model: PyTorch model (already loaded)
        tokenizer: Tokenizer (already loaded)
        test_names: List of test names to run
        device: Device string ('cuda' or 'cpu')
        limit: Optional limit on number of examples per test (for quick testing)
    
    Returns:
        dict: Results dictionary with metrics for each test
    
    Raises:
        RuntimeError: If benchmark execution fails
    """
    try:
        logger.info(f"Running benchmarks: {test_names}")
        logger.info(f"Using device: {device}")
        
        # Create custom model wrapper
        lm = CustomHFLM(model, tokenizer, device=device)
        
        # Prepare task list - simple_evaluate expects a list, not a comma-separated string
        tasks = test_names  # Pass as list directly
        
        # Run evaluation
        logger.info("Starting evaluation...")
        results = simple_evaluate(
            model=lm,
            tasks=tasks,
            limit=limit,  # None means full dataset
            device=device,
            batch_size=8,  # Adjust based on GPU memory
        )
        
        logger.info("Evaluation completed successfully")
        
        # Extract and format results
        formatted_results = {}
        for task_name in test_names:
            if task_name in results.get("results", {}):
                task_results = results["results"][task_name]
                formatted_results[task_name] = task_results
                logger.info(f"{task_name}: {task_results}")
            else:
                logger.warning(f"No results found for {task_name}")
                formatted_results[task_name] = {}
        
        return formatted_results
    
    except Exception as e:
        logger.error(f"Benchmark execution failed: {e}")
        raise RuntimeError(f"Failed to run benchmarks: {e}") from e


def run_single_test(model, tokenizer, test_name, device="cuda", limit=10):
    """
    Run a single test for quick validation.
    
    Args:
        model: PyTorch model
        tokenizer: Tokenizer
        test_name: Name of test to run
        device: Device string
        limit: Number of examples to test (default 10 for quick test)
    
    Returns:
        dict: Results for the test
    """
    logger.info(f"Running single test: {test_name} (limit={limit})")
    return run_benchmarks(model, tokenizer, [test_name], device=device, limit=limit)

