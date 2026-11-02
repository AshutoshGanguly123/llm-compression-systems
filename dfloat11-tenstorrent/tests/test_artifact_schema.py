"""Test compressed artifact schema and file format."""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from df11_tt.compress.compressor import compress_weight
from df11_tt.compress.artifact import CompressionArtifact, load_artifact


class TestArtifactSchema:
    @pytest.fixture
    def sample_artifact(self, tmp_path):
        torch.manual_seed(42)
        t = torch.randn(256, 512, dtype=torch.bfloat16)
        compress_weight(t, "test_schema_layer", str(tmp_path), verbose=False)
        return tmp_path / "test_schema_layer"

    def test_artifact_files_exist(self, sample_artifact):
        """Verify all required artifact files are created."""
        assert (sample_artifact / "metadata.json").exists()
        assert (sample_artifact / "exponent_bitstream.bin").exists()
        assert (sample_artifact / "sign_mantissa.npy").exists()
        assert (sample_artifact / "codec.json").exists()

    def test_metadata_fields(self, sample_artifact):
        """Verify metadata JSON has all required fields."""
        with open(sample_artifact / "metadata.json") as f:
            meta = json.load(f)

        required_fields = [
            "layer_name", "shape", "dtype", "original_bytes",
            "compressed_bytes", "compression_ratio", "checksum",
            "num_elements", "num_bits_exponent", "codec_version",
        ]
        for field in required_fields:
            assert field in meta, f"Missing field: {field}"

        assert meta["dtype"] == "bfloat16"
        assert meta["shape"] == [256, 512]
        assert meta["num_elements"] == 256 * 512
        assert meta["original_bytes"] == 256 * 512 * 2
        assert meta["compressed_bytes"] < meta["original_bytes"]
        assert meta["compression_ratio"] > 1.0
        assert len(meta["checksum"]) == 64  # SHA-256 hex

    def test_codec_json_fields(self, sample_artifact):
        """Verify codec JSON has required fields."""
        with open(sample_artifact / "codec.json") as f:
            codec = json.load(f)

        assert "code_table" in codec
        assert "symbol_counts" in codec
        assert len(codec["code_table"]) > 0

    def test_sign_mantissa_shape(self, sample_artifact):
        """Verify sign+mantissa payload has correct shape."""
        sm = np.load(str(sample_artifact / "sign_mantissa.npy"))
        assert sm.shape == (256 * 512,)
        assert sm.dtype == np.uint8

    def test_artifact_roundtrip_serialization(self):
        """Test CompressionArtifact to_dict/from_dict roundtrip."""
        artifact = CompressionArtifact(
            layer_name="test",
            shape=(128, 256),
            dtype="bfloat16",
            original_bytes=65536,
            compressed_bytes=45000,
            compression_ratio=1.456,
            checksum="abc123",
            num_elements=32768,
            num_bits_exponent=100000,
        )
        d = artifact.to_dict()
        restored = CompressionArtifact.from_dict(d)
        assert restored.layer_name == artifact.layer_name
        assert restored.shape == artifact.shape
        assert restored.original_bytes == artifact.original_bytes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
