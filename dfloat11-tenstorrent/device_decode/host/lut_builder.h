#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// Each LUT entry packs: [symbol (8 bits)] [code_length (8 bits)] into a uint16_t.
// The kernel reads 16 bits per entry. symbol in high byte, length in low byte.
struct DecodeLutEntry {
    uint8_t symbol;
    uint8_t code_length;
};

struct DecodeLut {
    std::vector<uint16_t> table;  // packed entries: (symbol << 8) | code_length
    uint32_t max_code_len;
    uint32_t table_size;          // = 1 << max_code_len
};

inline DecodeLut build_decode_lut(const std::unordered_map<uint32_t, std::string>& code_table) {
    if (code_table.empty()) {
        throw std::runtime_error("Empty code table");
    }

    uint32_t max_len = 0;
    for (const auto& [sym, bits] : code_table) {
        if (bits.size() > max_len) {
            max_len = static_cast<uint32_t>(bits.size());
        }
    }

    // Cap at 16 bits to keep LUT manageable (64K entries * 2B = 128KB)
    if (max_len > 16) {
        throw std::runtime_error("Huffman code length exceeds 16 bits, LUT too large");
    }

    uint32_t table_size = 1u << max_len;
    std::vector<uint16_t> table(table_size, 0);

    for (const auto& [sym, bits] : code_table) {
        uint32_t code_len = static_cast<uint32_t>(bits.size());
        uint32_t code_val = 0;
        for (char c : bits) {
            code_val = (code_val << 1) | (c - '0');
        }

        // This code matches any entry where the top code_len bits equal code_val.
        // Fill all entries where the suffix (max_len - code_len) bits vary.
        uint32_t suffix_bits = max_len - code_len;
        uint32_t num_entries = 1u << suffix_bits;
        uint16_t packed = (static_cast<uint16_t>(sym & 0xFF) << 8) | static_cast<uint16_t>(code_len & 0xFF);

        for (uint32_t suffix = 0; suffix < num_entries; suffix++) {
            uint32_t index = (code_val << suffix_bits) | suffix;
            table[index] = packed;
        }
    }

    DecodeLut lut;
    lut.table = std::move(table);
    lut.max_code_len = max_len;
    lut.table_size = table_size;
    return lut;
}

// CPU-side Huffman decode using the same LUT, for verification
inline std::vector<uint8_t> cpu_lut_decode(
    const DecodeLut& lut,
    const uint8_t* bitstream,
    uint64_t num_bits,
    uint32_t num_symbols
) {
    std::vector<uint8_t> output(num_symbols);
    uint64_t bit_pos = 0;
    uint32_t max_len = lut.max_code_len;
    uint32_t mask = lut.table_size - 1;

    for (uint32_t i = 0; i < num_symbols; i++) {
        // Peek max_len bits from the bitstream at bit_pos
        uint32_t peek = 0;
        for (uint32_t b = 0; b < max_len && (bit_pos + b) < num_bits; b++) {
            uint64_t byte_idx = (bit_pos + b) / 8;
            uint32_t bit_idx = 7 - ((bit_pos + b) % 8);  // MSB-first
            uint32_t bit = (bitstream[byte_idx] >> bit_idx) & 1;
            peek = (peek << 1) | bit;
        }
        // Pad with zeros if we're near the end
        uint32_t bits_available = static_cast<uint32_t>(
            (num_bits > bit_pos) ? (num_bits - bit_pos) : 0);
        if (bits_available < max_len) {
            peek <<= (max_len - bits_available);
        }

        uint16_t entry = lut.table[peek & mask];
        output[i] = static_cast<uint8_t>(entry >> 8);
        uint8_t code_len = static_cast<uint8_t>(entry & 0xFF);
        bit_pos += code_len;
    }

    return output;
}

// Full CPU-side BF16 reconstruction for verification
inline std::vector<uint16_t> cpu_reconstruct_bf16(
    const std::vector<uint8_t>& exponents,
    const std::vector<uint8_t>& sign_mantissa,
    uint32_t num_elements
) {
    std::vector<uint16_t> bf16(num_elements);
    for (uint32_t i = 0; i < num_elements; i++) {
        uint8_t sm = sign_mantissa[i];
        uint8_t sign = (sm >> 7) & 0x1;
        uint8_t mantissa = sm & 0x7F;
        uint8_t exponent = exponents[i];
        bf16[i] = (static_cast<uint16_t>(sign) << 15) |
                  (static_cast<uint16_t>(exponent) << 7) |
                  static_cast<uint16_t>(mantissa);
    }
    return bf16;
}
