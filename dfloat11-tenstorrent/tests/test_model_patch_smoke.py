"""Smoke test for model patching utilities."""

import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from df11_tt.runtime.model_patch import get_patchable_layers, extract_and_compress_layers, patch_model_layer


class TestModelPatch:
    def test_get_patchable_layers(self):
        """Identify patchable layers from a state dict."""
        state_dict = {
            "lm_head.weight": torch.randn(1000, 256, dtype=torch.bfloat16),
            "model.layers.0.mlp.gate_proj.weight": torch.randn(512, 256, dtype=torch.bfloat16),
            "model.layers.0.norm.weight": torch.randn(256, dtype=torch.bfloat16),
            "model.embed_tokens.weight": torch.randn(1000, 256, dtype=torch.bfloat16),
        }
        patchable = get_patchable_layers(state_dict, targets=["lm_head"])
        assert "lm_head.weight" in patchable
        assert len(patchable) == 1

    def test_get_all_patchable_layers(self):
        state_dict = {
            "lm_head.weight": torch.randn(1000, 256, dtype=torch.bfloat16),
            "model.layers.0.mlp.gate_proj.weight": torch.randn(512, 256, dtype=torch.bfloat16),
            "model.layers.0.norm.weight": torch.randn(256, dtype=torch.bfloat16),
        }
        patchable = get_patchable_layers(state_dict)
        assert len(patchable) == 2  # only 2D tensors

    def test_extract_and_compress(self, tmp_path):
        """Extract and compress layers from state dict."""
        state_dict = {
            "lm_head.weight": torch.randn(128, 64, dtype=torch.bfloat16),
        }
        artifact_map = extract_and_compress_layers(
            state_dict, ["lm_head.weight"], str(tmp_path), verbose=False,
        )
        assert "lm_head.weight" in artifact_map
        assert Path(artifact_map["lm_head.weight"]).exists()

    def test_patch_creates_runtime(self, tmp_path):
        """patch_model_layer should return a CompressedLinearRuntime."""
        from df11_tt.compress.compressor import compress_weight
        weight = torch.randn(64, 32, dtype=torch.bfloat16)
        compress_weight(weight, "patch_test", str(tmp_path), verbose=False)

        runtime = patch_model_layer(
            artifact_path=str(tmp_path / "patch_test"),
            device=None,
            verbose=False,
        )
        x = torch.randn(1, 16, 32, dtype=torch.bfloat16)
        output = runtime.forward(x)
        assert output.shape == (1, 16, 64)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
