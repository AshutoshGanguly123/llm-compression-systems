"""
Pre-download the model on login node (with internet) so GPU nodes can use cached version.
Run this on a login node before running the experiment on GPU nodes.
"""

import os
import sys
import logging
from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get model name from config
from config import MODEL_NAME

def download_model():
    """Download model and tokenizer to cache."""
    logger.info("=" * 80)
    logger.info("Pre-downloading model for offline use on GPU nodes")
    logger.info("=" * 80)
    logger.info(f"Model: {MODEL_NAME}")
    logger.info("")
    
    # Get token
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not hf_token:
        logger.warning("No HF_TOKEN found. Trying to read from ~/.huggingface/token")
        token_path = os.path.expanduser("~/.huggingface/token")
        if os.path.exists(token_path):
            with open(token_path, "r") as f:
                hf_token = f.read().strip()
            logger.info("Token read from file")
        else:
            logger.error("No token found! Please set HF_TOKEN or create ~/.huggingface/token")
            sys.exit(1)
    
    logger.info(f"Using token: {hf_token[:15]}...")
    logger.info("")
    
    try:
        # Download ALL files (tokenizer, model, config) WITHOUT loading into memory
        logger.info("Downloading model repository files (this may take a while)...")
        logger.info("Note: Only downloading files to disk, NOT loading into memory")
        logger.info("")
        
        # Use snapshot_download to download all files without loading model
        logger.info("Downloading files from HuggingFace...")
        cache_dir = snapshot_download(
            repo_id=MODEL_NAME,
            token=hf_token,
            local_files_only=False,  # Download if not cached
            ignore_patterns=["*.msgpack", "*.h5"],  # Skip unnecessary files
        )
        
        logger.info("")
        logger.info(f"✓ Model files downloaded to cache: {cache_dir}")
        logger.info("✓ All files are cached and ready for GPU node use")
        logger.info("✓ No model was loaded into memory - only files downloaded")
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("✓ Model successfully downloaded and cached!")
        logger.info("=" * 80)
        logger.info("")
        logger.info("The model is now cached and can be loaded on GPU nodes.")
        logger.info("You can now run: python3 experiment_runner.py")
        logger.info("")
        
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    download_model()

