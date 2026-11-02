"""Smoke test: verify TT linear works with decoded weights.

This test requires a TT device to be available.
Skip gracefully if no device is present.
"""

import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from df11_tt.compress.compressor import compress_weight
from df11_tt.runtime.decoder import decode_artifact

try:
    import ttnn
    TT_AVAILABLE = True
except ImportError:
    TT_AVAILABLE = False


@pytest.mark.skipif(not TT_AVAILABLE, reason="ttnn not available")
class TestTTLinearSmoke:
    def test_decoded_weight_on_tt_device(self, tmp_path):
        """Upload decoded weight to TT and run ttnn.linear."""
        torch.manual_seed(42)
        in_features, out_features = 2048, 2048
        weight = torch.randn(out_features, in_features, dtype=torch.bfloat16) * 0.02

        compress_weight(weight, "tt_smoke", str(tmp_path), verbose=False)
        decoded = decode_artifact(str(tmp_path / "tt_smoke"), verify=True, verbose=False)
        weight_T = decoded.t().contiguous()

        device = ttnn.open_device(device_id=0)
        try:
            tt_weight = ttnn.from_torch(
                weight_T, device=device, dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            activation = torch.randn(1, 32, in_features, dtype=torch.bfloat16) * 0.1
            tt_act = ttnn.from_torch(
                activation, device=device, dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            output = ttnn.linear(tt_act, tt_weight)
            ttnn.synchronize_device(device)

            result = ttnn.to_torch(output)
            assert result.shape[-1] == out_features

            ttnn.deallocate(tt_weight)
            ttnn.deallocate(tt_act)
            ttnn.deallocate(output)
        finally:
            ttnn.close_device(device)

    def test_baseline_vs_compressed_tt_output(self, tmp_path):
        """Baseline and compressed TT outputs should be identical
        (same weights, same compute path)."""
        torch.manual_seed(42)
        in_features, out_features = 256, 256
        weight = torch.randn(out_features, in_features, dtype=torch.bfloat16) * 0.02
        activation = torch.randn(1, 32, in_features, dtype=torch.bfloat16) * 0.1

        compress_weight(weight, "compare_test", str(tmp_path), verbose=False)
        decoded = decode_artifact(str(tmp_path / "compare_test"), verify=True, verbose=False)

        device = ttnn.open_device(device_id=0)
        try:
            tt_act = ttnn.from_torch(
                activation, device=device, dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            # Baseline
            tt_w_baseline = ttnn.from_torch(
                weight.t().contiguous(), device=device, dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            out_baseline = ttnn.linear(tt_act, tt_w_baseline)
            ttnn.synchronize_device(device)
            result_baseline = ttnn.to_torch(out_baseline)

            # Compressed
            tt_w_compressed = ttnn.from_torch(
                decoded.t().contiguous(), device=device, dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            out_compressed = ttnn.linear(tt_act, tt_w_compressed)
            ttnn.synchronize_device(device)
            result_compressed = ttnn.to_torch(out_compressed)

            assert torch.equal(result_baseline, result_compressed), \
                "Baseline and compressed TT outputs differ"

            ttnn.deallocate(tt_w_baseline)
            ttnn.deallocate(tt_w_compressed)
            ttnn.deallocate(tt_act)
            ttnn.deallocate(out_baseline)
            ttnn.deallocate(out_compressed)
        finally:
            ttnn.close_device(device)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
