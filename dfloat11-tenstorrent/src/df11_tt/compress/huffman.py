"""Huffman coding for BF16 exponent stream compression.

DFloat11 exploits the low-entropy structure of BF16 exponents:
most weights cluster around a few exponent values. Huffman coding
assigns shorter bit strings to frequent exponents.
"""

import heapq
from collections import Counter
from typing import Dict, Tuple

import numpy as np


class _HuffmanNode:
    __slots__ = ("symbol", "freq", "left", "right")

    def __init__(self, symbol=None, freq=0, left=None, right=None):
        self.symbol = symbol
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq


class HuffmanCodec:
    """Canonical Huffman codec for uint8 symbol streams."""

    def __init__(self):
        self.code_table: Dict[int, str] = {}
        self.decode_table: Dict[str, int] = {}
        self.symbol_counts: Dict[int, int] = {}

    def build_from_data(self, data: np.ndarray) -> "HuffmanCodec":
        """Build Huffman tree from a uint8 data array."""
        counts = Counter(data.tolist())
        self.symbol_counts = dict(counts)

        if len(counts) == 0:
            return self
        if len(counts) == 1:
            sym = list(counts.keys())[0]
            self.code_table = {sym: "0"}
            self.decode_table = {"0": sym}
            return self

        heap = [_HuffmanNode(symbol=s, freq=f) for s, f in counts.items()]
        heapq.heapify(heap)

        while len(heap) > 1:
            lo = heapq.heappop(heap)
            hi = heapq.heappop(heap)
            merged = _HuffmanNode(freq=lo.freq + hi.freq, left=lo, right=hi)
            heapq.heappush(heap, merged)

        root = heap[0]
        self.code_table = {}
        self._walk(root, "")
        self.decode_table = {v: k for k, v in self.code_table.items()}
        return self

    def _walk(self, node: _HuffmanNode, prefix: str):
        if node.symbol is not None:
            self.code_table[node.symbol] = prefix if prefix else "0"
            return
        if node.left:
            self._walk(node.left, prefix + "0")
        if node.right:
            self._walk(node.right, prefix + "1")

    def encode(self, data: np.ndarray) -> Tuple[bytes, int]:
        """Encode uint8 array into compressed bitstream.

        Returns:
            (compressed_bytes, num_bits): the packed bytes and total number of valid bits.
        """
        bits = []
        for val in data:
            bits.append(self.code_table[int(val)])
        bitstring = "".join(bits)
        num_bits = len(bitstring)

        padding = (8 - num_bits % 8) % 8
        bitstring += "0" * padding

        byte_array = bytearray()
        for i in range(0, len(bitstring), 8):
            byte_array.append(int(bitstring[i:i + 8], 2))

        return bytes(byte_array), num_bits

    def decode(self, data: bytes, num_bits: int, num_symbols: int) -> np.ndarray:
        """Decode compressed bitstream back to uint8 array.

        Args:
            data: compressed bytes
            num_bits: number of valid bits in the stream
            num_symbols: expected number of output symbols

        Returns:
            numpy uint8 array of decoded symbols
        """
        bitstring = "".join(f"{byte:08b}" for byte in data)[:num_bits]

        result = []
        current = ""
        for bit in bitstring:
            current += bit
            if current in self.decode_table:
                result.append(self.decode_table[current])
                current = ""
                if len(result) == num_symbols:
                    break

        return np.array(result, dtype=np.uint8)

    def to_dict(self) -> dict:
        """Serialize codec state to a JSON-compatible dict."""
        return {
            "code_table": {str(k): v for k, v in self.code_table.items()},
            "symbol_counts": {str(k): v for k, v in self.symbol_counts.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HuffmanCodec":
        """Reconstruct codec from serialized dict."""
        codec = cls()
        codec.code_table = {int(k): v for k, v in d["code_table"].items()}
        codec.decode_table = {v: int(k) for k, v in d["code_table"].items()}
        codec.symbol_counts = {int(k): v for k, v in d["symbol_counts"].items()}
        return codec

    def stats(self, data: np.ndarray) -> dict:
        """Compute compression statistics for a given data array."""
        original_bits = len(data) * 8
        _, compressed_bits = self.encode(data)
        return {
            "num_symbols": len(data),
            "unique_symbols": len(self.code_table),
            "original_bits": original_bits,
            "compressed_bits": compressed_bits,
            "compression_ratio_exponent": original_bits / max(compressed_bits, 1),
        }
