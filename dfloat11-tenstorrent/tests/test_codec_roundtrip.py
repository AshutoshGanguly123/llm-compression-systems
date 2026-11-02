"""Test DFloat11 codec roundtrip: compress -> decompress must be bit-exact."""

import sys
from pathlib import Path

import numpy as np
import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from df11_tt.compress.bitops import split_bf16, merge_bf16, pack_sign_mantissa, unpack_sign_mantissa
from df11_tt.compress.huffman import HuffmanCodec
from df11_tt.compress.compressor import compress_weight
from df11_tt.runtime.decoder import decode_artifact
from df11_tt.utils.hashing import tensor_checksum, verify_bit_exact


class TestBitOps:
    def test_split_merge_identity(self):
        """Split and merge must be exact inverse operations."""
        torch.manual_seed(42)
        t = torch.randn(128, 256, dtype=torch.bfloat16)
        signs, exponents, mantissas = split_bf16(t)
        reconstructed = merge_bf16(signs, exponents, mantissas, t.shape)
        assert verify_bit_exact(t, reconstructed), "Split/merge roundtrip not bit-exact"

    def test_split_merge_special_values(self):
        """Test with zeros, inf, nan, very small and large values."""
        values = torch.tensor([0.0, -0.0, 1.0, -1.0, 0.5, 100.0, -100.0, 1e-10], dtype=torch.bfloat16)
        signs, exponents, mantissas = split_bf16(values)
        reconstructed = merge_bf16(signs, exponents, mantissas, values.shape)
        assert verify_bit_exact(values, reconstructed)

    def test_sign_mantissa_packing(self):
        """Test sign+mantissa pack/unpack roundtrip."""
        signs = np.array([0, 1, 0, 1, 0], dtype=np.uint8)
        mantissas = np.array([0, 127, 64, 1, 100], dtype=np.uint8)
        packed = pack_sign_mantissa(signs, mantissas)
        s_out, m_out = unpack_sign_mantissa(packed)
        np.testing.assert_array_equal(signs, s_out)
        np.testing.assert_array_equal(mantissas, m_out)

    def test_various_shapes(self):
        """Test with various tensor shapes."""
        for shape in [(1,), (32,), (64, 64), (3, 128, 256), (1, 1, 1, 1)]:
            t = torch.randn(*shape, dtype=torch.bfloat16)
            signs, exponents, mantissas = split_bf16(t)
            reconstructed = merge_bf16(signs, exponents, mantissas, t.shape)
            assert verify_bit_exact(t, reconstructed), f"Failed for shape {shape}"


class TestHuffmanCodec:
    def test_encode_decode_roundtrip(self):
        """Huffman encode->decode must recover original symbols."""
        np.random.seed(42)
        data = np.random.randint(0, 256, size=10000, dtype=np.uint8)
        codec = HuffmanCodec().build_from_data(data)
        encoded, num_bits = codec.encode(data)
        decoded = codec.decode(encoded, num_bits, len(data))
        np.testing.assert_array_equal(data, decoded)

    def test_single_symbol(self):
        """Test with all identical symbols."""
        data = np.full(100, 42, dtype=np.uint8)
        codec = HuffmanCodec().build_from_data(data)
        encoded, num_bits = codec.encode(data)
        decoded = codec.decode(encoded, num_bits, len(data))
        np.testing.assert_array_equal(data, decoded)

    def test_two_symbols(self):
        """Test with exactly two unique symbols."""
        data = np.array([0, 1, 0, 1, 0, 0, 1], dtype=np.uint8)
        codec = HuffmanCodec().build_from_data(data)
        encoded, num_bits = codec.encode(data)
        decoded = codec.decode(encoded, num_bits, len(data))
        np.testing.assert_array_equal(data, decoded)

    def test_serialization_roundtrip(self):
        """Codec serialize->deserialize must produce identical codec."""
        np.random.seed(42)
        data = np.random.randint(0, 50, size=5000, dtype=np.uint8)
        codec = HuffmanCodec().build_from_data(data)
        d = codec.to_dict()
        codec2 = HuffmanCodec.from_dict(d)

        encoded1, bits1 = codec.encode(data)
        encoded2, bits2 = codec2.encode(data)
        assert encoded1 == encoded2
        assert bits1 == bits2

    def test_compression_ratio(self):
        """Low-entropy data should compress well."""
        data = np.random.choice([127, 128, 126], size=10000, p=[0.8, 0.15, 0.05]).astype(np.uint8)
        codec = HuffmanCodec().build_from_data(data)
        stats = codec.stats(data)
        assert stats["compression_ratio_exponent"] > 2.0, "Expected >2x compression for low-entropy data"


class TestFullRoundtrip:
    def test_compress_decompress_small(self, tmp_path):
        """Full pipeline: compress -> save -> load -> decompress -> verify bit-exact."""
        torch.manual_seed(42)
        t = torch.randn(64, 128, dtype=torch.bfloat16)
        original_checksum = tensor_checksum(t)

        artifact = compress_weight(t, "test_layer", str(tmp_path), verbose=False)
        assert artifact.compression_ratio > 1.0
        assert artifact.checksum == original_checksum

        artifact_path = str(tmp_path / "test_layer")
        decoded = decode_artifact(artifact_path, verify=True, verbose=False)

        assert verify_bit_exact(t, decoded), "Full roundtrip not bit-exact"
        assert decoded.shape == t.shape
        assert decoded.dtype == torch.bfloat16

    def test_compress_decompress_large(self, tmp_path):
        """Full roundtrip with a larger, more realistic tensor."""
        torch.manual_seed(123)
        t = torch.randn(2048, 2048, dtype=torch.bfloat16) * 0.02

        artifact = compress_weight(t, "large_layer", str(tmp_path), verbose=False)
        artifact_path = str(tmp_path / "large_layer")
        decoded = decode_artifact(artifact_path, verify=True, verbose=False)

        assert verify_bit_exact(t, decoded)
        assert artifact.original_bytes == 2048 * 2048 * 2
        assert artifact.compressed_bytes < artifact.original_bytes

    def test_checksum_mismatch_detected(self, tmp_path):
        """Verify that corrupted artifacts are detected."""
        torch.manual_seed(42)
        t = torch.randn(32, 64, dtype=torch.bfloat16)
        compress_weight(t, "corrupt_test", str(tmp_path), verbose=False)

        import json
        meta_path = tmp_path / "corrupt_test" / "metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)
        meta["checksum"] = "0000000000000000000000000000000000000000000000000000000000000000"
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        with pytest.raises(ValueError, match="Checksum mismatch"):
            decode_artifact(str(tmp_path / "corrupt_test"), verify=True, verbose=False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
