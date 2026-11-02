#!/usr/bin/env bash
# Copy to setup_hf_token.sh locally (gitignored) or export in your shell.
# Do not commit real tokens.
export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN to a Hugging Face access token}"
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential || true
