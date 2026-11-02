// DFloat11 Huffman Decode Kernel for Tensix RISC-V
//
// Runs on a single data-movement RISC-V core. Reads compressed exponent
// bitstream + packed sign/mantissa from device DRAM, decodes exponents via
// a flat LUT, reassembles BF16 words, and writes the result back to DRAM.
//
// Memory layout passed via runtime args:
//   arg0: lut_dram_addr        - DRAM address of decode LUT (uint16_t[2^max_code_len])
//   arg1: lut_l1_addr          - L1 scratch address for LUT
//   arg2: bitstream_dram_addr  - DRAM address of exponent bitstream
//   arg3: bitstream_l1_addr    - L1 scratch address for bitstream chunk
//   arg4: signmant_dram_addr   - DRAM address of sign/mantissa array
//   arg5: signmant_l1_addr     - L1 scratch address for sign/mantissa chunk
//   arg6: output_dram_addr     - DRAM address of output BF16 buffer
//   arg7: output_l1_addr       - L1 scratch address for output chunk
//   arg8: num_elements         - total number of BF16 elements to decode
//   arg9: lut_size_bytes       - size of LUT in bytes
//   arg10: bitstream_size_bytes - total size of exponent bitstream
//   arg11: max_code_len        - maximum Huffman code length (for LUT indexing)
//   arg12: chunk_elements      - number of elements to process per chunk

void kernel_main() {
    uint32_t lut_dram_addr       = get_arg_val<uint32_t>(0);
    uint32_t lut_l1_addr         = get_arg_val<uint32_t>(1);
    uint32_t bitstream_dram_addr = get_arg_val<uint32_t>(2);
    uint32_t bitstream_l1_addr   = get_arg_val<uint32_t>(3);
    uint32_t signmant_dram_addr  = get_arg_val<uint32_t>(4);
    uint32_t signmant_l1_addr    = get_arg_val<uint32_t>(5);
    uint32_t output_dram_addr    = get_arg_val<uint32_t>(6);
    uint32_t output_l1_addr      = get_arg_val<uint32_t>(7);
    uint32_t num_elements        = get_arg_val<uint32_t>(8);
    uint32_t lut_size_bytes      = get_arg_val<uint32_t>(9);
    uint32_t bitstream_total_bytes = get_arg_val<uint32_t>(10);
    uint32_t max_code_len        = get_arg_val<uint32_t>(11);
    uint32_t chunk_elements      = get_arg_val<uint32_t>(12);

    // Address generators for interleaved DRAM buffers
    InterleavedAddrGen<true> lut_gen       = {.bank_base_address = lut_dram_addr,       .page_size = lut_size_bytes};
    InterleavedAddrGen<true> output_gen    = {.bank_base_address = output_dram_addr,    .page_size = bitstream_total_bytes};

    // Step 1: Load the full decode LUT into L1 (small: e.g. 16KB for 13-bit codes)
    uint64_t lut_noc_addr = get_noc_addr(0, lut_gen);
    noc_async_read(lut_noc_addr, lut_l1_addr, lut_size_bytes);
    noc_async_read_barrier();

    volatile uint16_t* lut = reinterpret_cast<volatile uint16_t*>(lut_l1_addr);
    uint32_t lut_mask = (1u << max_code_len) - 1;

    // Process elements in chunks
    uint64_t global_bit_pos = 0;
    uint32_t elements_done = 0;

    while (elements_done < num_elements) {
        uint32_t this_chunk = chunk_elements;
        if (elements_done + this_chunk > num_elements) {
            this_chunk = num_elements - elements_done;
        }

        // Calculate how many bitstream bytes we need for this chunk.
        // Worst case: each symbol uses max_code_len bits.
        // We read from the current byte offset, enough bytes to cover worst case + alignment.
        uint32_t start_byte = static_cast<uint32_t>(global_bit_pos / 8);
        uint32_t worst_case_bits = this_chunk * max_code_len + max_code_len;
        uint32_t worst_case_bytes = (worst_case_bits + 7) / 8;
        if (start_byte + worst_case_bytes > bitstream_total_bytes) {
            worst_case_bytes = bitstream_total_bytes - start_byte;
        }
        // Align read size to 64 bytes for Blackhole NoC alignment
        uint32_t aligned_bytes = ((worst_case_bytes + 63) / 64) * 64;
        if (start_byte + aligned_bytes > ((bitstream_total_bytes + 63) / 64) * 64) {
            aligned_bytes = ((bitstream_total_bytes - start_byte + 63) / 64) * 64;
        }

        // Align start_byte down to 64B boundary for DRAM read alignment
        uint32_t aligned_start = (start_byte / 64) * 64;
        uint32_t start_offset = start_byte - aligned_start;
        uint32_t read_bytes = ((start_offset + worst_case_bytes + 63) / 64) * 64;

        // Read bitstream chunk into L1
        uint64_t bs_noc_addr = get_noc_addr(0, InterleavedAddrGen<true>{
            .bank_base_address = bitstream_dram_addr, .page_size = bitstream_total_bytes});
        // Manually compute the address offset
        // For single-page interleaved buffers, the entire buffer is one page at bank 0
        // We use raw noc address + offset
        noc_async_read(bs_noc_addr + aligned_start, bitstream_l1_addr, read_bytes);

        // Read sign/mantissa chunk into L1 (1 byte per element, linear)
        uint32_t sm_start = elements_done;
        uint32_t sm_bytes = this_chunk;
        uint32_t sm_aligned_start = (sm_start / 64) * 64;
        uint32_t sm_offset = sm_start - sm_aligned_start;
        uint32_t sm_read_bytes = ((sm_offset + sm_bytes + 63) / 64) * 64;

        uint64_t sm_noc_addr = get_noc_addr(0, InterleavedAddrGen<true>{
            .bank_base_address = signmant_dram_addr, .page_size = bitstream_total_bytes});
        noc_async_read(sm_noc_addr + sm_aligned_start, signmant_l1_addr, sm_read_bytes);

        noc_async_read_barrier();

        // Decode this chunk
        volatile uint8_t* bs_data = reinterpret_cast<volatile uint8_t*>(bitstream_l1_addr);
        volatile uint8_t* sm_data = reinterpret_cast<volatile uint8_t*>(signmant_l1_addr);
        volatile uint16_t* out_data = reinterpret_cast<volatile uint16_t*>(output_l1_addr);

        uint32_t local_bit_pos = static_cast<uint32_t>(global_bit_pos - static_cast<uint64_t>(aligned_start) * 8);

        for (uint32_t i = 0; i < this_chunk; i++) {
            // Peek max_code_len bits from the bitstream (MSB-first)
            uint32_t peek = 0;
            uint32_t bpos = local_bit_pos;
            for (uint32_t b = 0; b < max_code_len; b++) {
                uint32_t byte_idx = bpos / 8;
                uint32_t bit_idx = 7 - (bpos % 8);
                uint32_t bit = (bs_data[byte_idx] >> bit_idx) & 1;
                peek = (peek << 1) | bit;
                bpos++;
            }

            // LUT lookup
            uint16_t entry = lut[peek & lut_mask];
            uint8_t exponent = static_cast<uint8_t>(entry >> 8);
            uint8_t code_len = static_cast<uint8_t>(entry & 0xFF);
            local_bit_pos += code_len;

            // Read sign+mantissa and reassemble BF16
            uint8_t sm_byte = sm_data[sm_offset + i];
            uint8_t sign = (sm_byte >> 7) & 0x1;
            uint8_t mantissa = sm_byte & 0x7F;

            uint16_t bf16_val = (static_cast<uint16_t>(sign) << 15) |
                                (static_cast<uint16_t>(exponent) << 7) |
                                static_cast<uint16_t>(mantissa);

            out_data[i] = bf16_val;
        }

        // Update global bit position
        global_bit_pos = static_cast<uint64_t>(aligned_start) * 8 + local_bit_pos;

        // Write decoded BF16 chunk to output DRAM
        uint32_t out_offset = elements_done * 2;  // 2 bytes per BF16
        uint32_t out_bytes = this_chunk * 2;
        // Align write to 16 bytes (Blackhole NoC write alignment)
        uint32_t aligned_out_bytes = ((out_bytes + 15) / 16) * 16;

        uint64_t out_noc_addr = get_noc_addr(0, InterleavedAddrGen<true>{
            .bank_base_address = output_dram_addr, .page_size = num_elements * 2});
        noc_async_write(output_l1_addr, out_noc_addr + out_offset, aligned_out_bytes);
        noc_async_write_barrier();

        elements_done += this_chunk;
    }
}
