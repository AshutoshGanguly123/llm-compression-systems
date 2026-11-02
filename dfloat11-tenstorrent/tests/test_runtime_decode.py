"""Test runtime decode: verify the decoder produces correct tensors."""

import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from df11_tt.compress.compressor import compress_weight
from df11_tt.runtime.decoder import decode_artifact
from df11_tt.runtime.compressed_linear import CompressedLinearRuntime
from df11_tt.utils.hashing import verify_bit_exact, tensor_checksum


class TestRuntimeDecode:
    def test_decode_produces_correct_shape(self, tmp_path):
        """Decoded tensor must have original shape."""
        torch.manual_seed(42)
        t = torch.randn(128, 256, dtype=torch.bfloat16)
        compress_weight(t, "shape_test", str(tmp_path), verbose=False)
        decoded = decode_artifact(str(tmp_path / "shape_test"), verbose=False)
        assert decoded.shape == (128, 256)
        assert decoded.dtype == torch.bfloat16

    def test_decode_produces_correct_dtype(self, tmp_path):
        torch.manual_seed(42)
        t = torch.randn(64, 64, dtype=torch.bfloat16)
        compress_weight(t, "dtype_test", str(tmp_path), verbose=False)
        decoded = decode_artifact(str(tmp_path / "dtype_test"), verbose=False)
        assert decoded.dtype == torch.bfloat16

    def test_decode_bit_exact(self, tmp_path):
        """Decoded tensor must be bitwise identical to original."""
        torch.manual_seed(42)
        t = torch.randn(256, 512, dtype=torch.bfloat16)
        compress_weight(t, "exact_test", str(tmp_path), verbose=False)
        decoded = decode_artifact(str(tmp_path / "exact_test"), verify=True, verbose=False)
        assert verify_bit_exact(t, decoded)

    def test_decode_with_verify_flag(self, tmp_path):
        """Verify flag should trigger checksum validation."""
        torch.manual_seed(42)
        t = torch.randn(32, 64, dtype=torch.bfloat16)
        compress_weight(t, "verify_test", str(tmp_path), verbose=False)
        decoded = decode_artifact(str(tmp_path / "verify_test"), verify=True, verbose=False)
        assert decoded is not None


class TestCompressedLinearCPU:
    def test_cpu_forward(self, tmp_path):
        """CompressedLinearRuntime CPU forward should produce valid output."""
        torch.manual_seed(42)
        weight = torch.randn(64, 128, dtype=torch.bfloat16)
        compress_weight(weight, "linear_test", str(tmp_path), verbose=False)

        runtime = CompressedLinearRuntime(
            artifact_path=str(tmp_path / "linear_test"),
            device=None,
            verbose=False,
        )
        x = torch.randn(1, 32, 128, dtype=torch.bfloat16)
        output = runtime.forward(x)
        assert output.shape == (1, 32, 64)

    def test_cpu_forward_metrics(self, tmp_path):
        """CompressedLinearRuntime should record timing metrics."""
        torch.manual_seed(42)
        weight = torch.randn(64, 128, dtype=torch.bfloat16)
        compress_weight(weight, "metrics_test", str(tmp_path), verbose=False)

        runtime = CompressedLinearRuntime(
            artifact_path=str(tmp_path / "metrics_test"),
            device=None,
            verbose=False,
        )
        x = torch.randn(1, 32, 128, dtype=torch.bfloat16)
        runtime.forward(x)

        metrics = runtime.get_metrics()
        assert "decode_ms" in metrics
        assert "tt_op_ms" in metrics
        assert "total_layer_ms" in metrics
        assert all(v > 0 for v in metrics.values())

    def test_cached_mode(self, tmp_path):
        """Cached mode should reuse decoded weights."""
        torch.manual_seed(42)
        weight = torch.randn(64, 128, dtype=torch.bfloat16)
        compress_weight(weight, "cache_test", str(tmp_path), verbose=False)

        runtime = CompressedLinearRuntime(
            artifact_path=str(tmp_path / "cache_test"),
            device=None,
            cache_decoded=True,
            verbose=False,
        )
        x = torch.randn(1, 32, 128, dtype=torch.bfloat16)
        out1 = runtime.forward(x)
        out2 = runtime.forward(x)
        assert torch.equal(out1, out2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
