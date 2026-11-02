"""
Model loader for loading Llama models in FP16 precision.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_fp16(model_name, device="cuda"):
    """
    Load a model and tokenizer, converting model weights to FP16.
    
    Args:
        model_name: HuggingFace model identifier
        device: Device to load model on ('cuda' or 'cpu')
    
    Returns:
        tuple: (model, tokenizer)
    
    Raises:
        RuntimeError: If model loading fails
    """
    try:
        logger.info(f"Loading model: {model_name}")
        logger.info(f"Using device: {device}")
        
        # Verify GPU if using CUDA
        if device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA device requested but not available!")
            gpu_id = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(gpu_id)
            logger.info(f"GPU ID: {gpu_id}")
            logger.info(f"GPU Name: {gpu_name}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(gpu_id).total_memory / 1e9:.2f} GB")
        
        # Get HuggingFace token from environment or use None
        import os
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token,
        )
        
        # Set pad token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        logger.info("Loading model...")
        # Load directly to GPU - avoid CPU to prevent system issues
        # Use single GPU device_map to load directly on GPU without CPU intermediate
        if device == "cuda":
            gpu_id = torch.cuda.current_device()
            device_map = f"cuda:{gpu_id}"  # Load directly to specific GPU
            logger.info(f"Loading directly to GPU {gpu_id} (no CPU intermediate)...")
        else:
            device_map = None
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,  # FP16 precision
            device_map=device_map,  # Load directly to GPU, no CPU intermediate
            low_cpu_mem_usage=True,  # Minimize CPU memory usage
            token=hf_token,  # Use HuggingFace token for gated models
        )
        
        # Verify model is on GPU and in FP16
        if device == "cuda":
            sample_param = next(model.parameters())
            if sample_param.device.type != "cuda":
                raise RuntimeError(f"Model not on GPU after device_map! Device: {sample_param.device}")
            # Ensure FP16 precision
            if sample_param.dtype != torch.float16:
                logger.info("Converting model to FP16...")
                model = model.half()
        
        # Verify model is in FP16
        sample_param = next(model.parameters())
        if sample_param.dtype != torch.float16:
            logger.warning(f"Model dtype is {sample_param.dtype}, expected float16. Converting...")
            model = model.half()
        
        logger.info("Model loaded successfully in FP16")
        sample_param = next(model.parameters())
        logger.info(f"Model dtype: {sample_param.dtype}")
        logger.info(f"Model device: {sample_param.device}")
        
        # Verify model is on GPU if requested
        if device == "cuda":
            if not sample_param.device.type == "cuda":
                raise RuntimeError(f"Model not on GPU! Device: {sample_param.device}")
            logger.info(f"✓ Model successfully loaded on GPU {torch.cuda.current_device()}")
        
        # Test forward pass
        logger.info("Testing forward pass...")
        model.eval()
        with torch.no_grad():
            # Ensure input is on the same device as model
            test_input = tokenizer("Hello", return_tensors="pt")
            # Move input to the same device as the first model parameter
            model_device = next(model.parameters()).device
            test_input = {k: v.to(model_device) for k, v in test_input.items()}
            _ = model(**test_input)
        logger.info("Forward pass test successful")
        
        return model, tokenizer
    
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Model loading failed: {e}") from e


def verify_model_fp16(model):
    """
    Verify that all model parameters are in FP16.
    
    Args:
        model: PyTorch model
    
    Returns:
        bool: True if all parameters are FP16
    """
    for name, param in model.named_parameters():
        if param.dtype != torch.float16:
            logger.warning(f"Parameter {name} is not FP16: {param.dtype}")
            return False
    return True

