#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/distributed.hpp>

#include "artifact_loader.h"
#include "lut_builder.h"

using namespace tt;
using namespace tt::tt_metal;

static constexpr uint32_t NOC_ALIGNMENT = 64;
static constexpr uint32_t CHUNK_ELEMENTS = 32768;

static uint32_t align_up(uint32_t val, uint32_t alignment) {
    return ((val + alignment - 1) / alignment) * alignment;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <artifact_dir> [output_file]" << std::endl;
        std::cerr << "  artifact_dir: path to compressed artifact directory" << std::endl;
        std::cerr << "  output_file:  optional path to write decoded BF16 binary" << std::endl;
        return 1;
    }

    std::string artifact_dir = argv[1];
    std::string output_file = (argc > 2) ? argv[2] : "";
    std::string kernel_path = "";

    // Locate the kernel file
    std::filesystem::path exe_dir = std::filesystem::path(argv[0]).parent_path();
    std::vector<std::string> kernel_search_paths = {
        (exe_dir / "../kernels/huffman_decode_kernel.cpp").string(),
        (exe_dir / "kernels/huffman_decode_kernel.cpp").string(),
        "device_decode/kernels/huffman_decode_kernel.cpp",
        "kernels/huffman_decode_kernel.cpp",
    };
    for (const auto& p : kernel_search_paths) {
        if (std::filesystem::exists(p)) {
            kernel_path = std::filesystem::canonical(p).string();
            break;
        }
    }
    if (kernel_path.empty()) {
        std::cerr << "ERROR: Cannot find huffman_decode_kernel.cpp" << std::endl;
        return 1;
    }

    auto t_start = std::chrono::high_resolution_clock::now();

    // ---- Step 1: Load compressed artifact ----
    std::cout << "=== DFloat11 Device-Side Huffman Decode ===" << std::endl;
    std::cout << "Loading artifact from: " << artifact_dir << std::endl;

    auto t_load_start = std::chrono::high_resolution_clock::now();
    CompressedArtifact artifact = load_compressed_artifact(artifact_dir);
    auto t_load_end = std::chrono::high_resolution_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(t_load_end - t_load_start).count();

    uint32_t num_elements = artifact.metadata.num_elements;
    uint64_t num_bits = artifact.metadata.num_bits_exponent;
    std::cout << "  Layer: " << artifact.metadata.layer_name << std::endl;
    std::cout << "  Elements: " << num_elements << std::endl;
    std::cout << "  Bitstream: " << artifact.exponent_bitstream.size() << " bytes" << std::endl;
    std::cout << "  Sign/mantissa: " << artifact.sign_mantissa.size() << " bytes" << std::endl;
    std::cout << "  Artifact load time: " << load_ms << " ms" << std::endl;

    // ---- Step 2: Build decode LUT ----
    auto t_lut_start = std::chrono::high_resolution_clock::now();
    DecodeLut lut = build_decode_lut(artifact.codec.code_table);
    auto t_lut_end = std::chrono::high_resolution_clock::now();
    double lut_ms = std::chrono::duration<double, std::milli>(t_lut_end - t_lut_start).count();

    std::cout << "  LUT: " << lut.table_size << " entries, max_code_len=" << lut.max_code_len
              << ", " << (lut.table_size * 2) << " bytes (" << lut_ms << " ms)" << std::endl;

    // ---- Step 3: CPU reference decode ----
    auto t_cpu_start = std::chrono::high_resolution_clock::now();
    std::vector<uint8_t> cpu_exponents = cpu_lut_decode(
        lut, artifact.exponent_bitstream.data(), num_bits, num_elements);
    std::vector<uint16_t> cpu_bf16 = cpu_reconstruct_bf16(
        cpu_exponents, artifact.sign_mantissa, num_elements);
    auto t_cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t_cpu_end - t_cpu_start).count();
    std::cout << "  CPU reference decode: " << cpu_ms << " ms" << std::endl;

    // ---- Step 4: Initialize TT device ----
    std::cout << "\nInitializing TT device..." << std::endl;
    auto t_device_start = std::chrono::high_resolution_clock::now();
    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(0);
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    auto t_device_end = std::chrono::high_resolution_clock::now();
    double device_ms = std::chrono::duration<double, std::milli>(t_device_end - t_device_start).count();
    std::cout << "  Device init: " << device_ms << " ms" << std::endl;

    // ---- Step 5: Prepare buffers ----
    // Pad all sizes to NOC alignment
    uint32_t lut_size_bytes = lut.table_size * sizeof(uint16_t);
    uint32_t lut_buf_size = align_up(lut_size_bytes, NOC_ALIGNMENT);

    uint32_t bitstream_bytes = static_cast<uint32_t>(artifact.exponent_bitstream.size());
    uint32_t bitstream_buf_size = align_up(bitstream_bytes, NOC_ALIGNMENT);

    uint32_t signmant_bytes = static_cast<uint32_t>(artifact.sign_mantissa.size());
    uint32_t signmant_buf_size = align_up(signmant_bytes, NOC_ALIGNMENT);

    uint32_t output_bytes = num_elements * 2;  // 2 bytes per BF16
    uint32_t output_buf_size = align_up(output_bytes, NOC_ALIGNMENT);

    // L1 scratch space sizing
    uint32_t chunk_elements = CHUNK_ELEMENTS;
    uint32_t bs_chunk_worst = align_up((chunk_elements * lut.max_code_len + lut.max_code_len + 7) / 8 + NOC_ALIGNMENT, NOC_ALIGNMENT);
    uint32_t sm_chunk_size = align_up(chunk_elements, NOC_ALIGNMENT);
    uint32_t out_chunk_size = align_up(chunk_elements * 2, NOC_ALIGNMENT);
    uint32_t total_l1_scratch = lut_buf_size + bs_chunk_worst + sm_chunk_size + out_chunk_size;

    std::cout << "  L1 scratch needed: " << total_l1_scratch << " bytes ("
              << (total_l1_scratch / 1024) << " KB)" << std::endl;

    // Allocate DRAM buffers (one page = entire buffer, interleaved)
    distributed::DeviceLocalBufferConfig dram_config_lut{
        .page_size = lut_buf_size, .buffer_type = BufferType::DRAM};
    distributed::DeviceLocalBufferConfig dram_config_bs{
        .page_size = bitstream_buf_size, .buffer_type = BufferType::DRAM};
    distributed::DeviceLocalBufferConfig dram_config_sm{
        .page_size = signmant_buf_size, .buffer_type = BufferType::DRAM};
    distributed::DeviceLocalBufferConfig dram_config_out{
        .page_size = output_buf_size, .buffer_type = BufferType::DRAM};

    distributed::ReplicatedBufferConfig rep_lut{.size = lut_buf_size};
    distributed::ReplicatedBufferConfig rep_bs{.size = bitstream_buf_size};
    distributed::ReplicatedBufferConfig rep_sm{.size = signmant_buf_size};
    distributed::ReplicatedBufferConfig rep_out{.size = output_buf_size};

    auto lut_dram_buf = distributed::MeshBuffer::create(rep_lut, dram_config_lut, mesh_device.get());
    auto bs_dram_buf = distributed::MeshBuffer::create(rep_bs, dram_config_bs, mesh_device.get());
    auto sm_dram_buf = distributed::MeshBuffer::create(rep_sm, dram_config_sm, mesh_device.get());
    auto out_dram_buf = distributed::MeshBuffer::create(rep_out, dram_config_out, mesh_device.get());

    // L1 scratch buffers
    distributed::DeviceLocalBufferConfig l1_config_lut{
        .page_size = lut_buf_size, .buffer_type = BufferType::L1};
    distributed::DeviceLocalBufferConfig l1_config_bs{
        .page_size = bs_chunk_worst, .buffer_type = BufferType::L1};
    distributed::DeviceLocalBufferConfig l1_config_sm{
        .page_size = sm_chunk_size, .buffer_type = BufferType::L1};
    distributed::DeviceLocalBufferConfig l1_config_out{
        .page_size = out_chunk_size, .buffer_type = BufferType::L1};

    distributed::ReplicatedBufferConfig rep_l1_lut{.size = lut_buf_size};
    distributed::ReplicatedBufferConfig rep_l1_bs{.size = bs_chunk_worst};
    distributed::ReplicatedBufferConfig rep_l1_sm{.size = sm_chunk_size};
    distributed::ReplicatedBufferConfig rep_l1_out{.size = out_chunk_size};

    auto lut_l1_buf = distributed::MeshBuffer::create(rep_l1_lut, l1_config_lut, mesh_device.get());
    auto bs_l1_buf = distributed::MeshBuffer::create(rep_l1_bs, l1_config_bs, mesh_device.get());
    auto sm_l1_buf = distributed::MeshBuffer::create(rep_l1_sm, l1_config_sm, mesh_device.get());
    auto out_l1_buf = distributed::MeshBuffer::create(rep_l1_out, l1_config_out, mesh_device.get());

    // ---- Step 6: Upload data to device DRAM ----
    std::cout << "Uploading data to device DRAM..." << std::endl;
    auto t_upload_start = std::chrono::high_resolution_clock::now();

    // Pack LUT as uint32_t vector (host API works with uint32_t)
    std::vector<uint32_t> lut_data_u32(lut_buf_size / sizeof(uint32_t), 0);
    std::memcpy(lut_data_u32.data(), lut.table.data(), lut_size_bytes);
    distributed::EnqueueWriteMeshBuffer(cq, lut_dram_buf, lut_data_u32, /*blocking=*/false);

    // Pack bitstream as uint32_t vector
    std::vector<uint32_t> bs_data_u32(bitstream_buf_size / sizeof(uint32_t), 0);
    std::memcpy(bs_data_u32.data(), artifact.exponent_bitstream.data(), bitstream_bytes);
    distributed::EnqueueWriteMeshBuffer(cq, bs_dram_buf, bs_data_u32, /*blocking=*/false);

    // Pack sign/mantissa as uint32_t vector
    std::vector<uint32_t> sm_data_u32(signmant_buf_size / sizeof(uint32_t), 0);
    std::memcpy(sm_data_u32.data(), artifact.sign_mantissa.data(), signmant_bytes);
    distributed::EnqueueWriteMeshBuffer(cq, sm_dram_buf, sm_data_u32, /*blocking=*/false);

    distributed::Finish(cq);
    auto t_upload_end = std::chrono::high_resolution_clock::now();
    double upload_ms = std::chrono::duration<double, std::milli>(t_upload_end - t_upload_start).count();
    std::cout << "  Upload time: " << upload_ms << " ms" << std::endl;

    // ---- Step 7: Create program and kernel ----
    std::cout << "Launching decode kernel on device..." << std::endl;
    std::cout << "  Kernel path: " << kernel_path << std::endl;

    Program program = CreateProgram();
    constexpr tt::tt_metal::CoreCoord core = {0, 0};

    KernelHandle kernel_id = CreateKernel(
        program,
        kernel_path,
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default});

    SetRuntimeArgs(
        program,
        kernel_id,
        core,
        {
            static_cast<uint32_t>(lut_dram_buf->address()),
            static_cast<uint32_t>(lut_l1_buf->address()),
            static_cast<uint32_t>(bs_dram_buf->address()),
            static_cast<uint32_t>(bs_l1_buf->address()),
            static_cast<uint32_t>(sm_dram_buf->address()),
            static_cast<uint32_t>(sm_l1_buf->address()),
            static_cast<uint32_t>(out_dram_buf->address()),
            static_cast<uint32_t>(out_l1_buf->address()),
            num_elements,
            lut_buf_size,
            bitstream_buf_size,
            lut.max_code_len,
            chunk_elements,
        });

    // ---- Step 8: Execute kernel ----
    auto t_kernel_start = std::chrono::high_resolution_clock::now();

    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
    distributed::Finish(cq);

    auto t_kernel_end = std::chrono::high_resolution_clock::now();
    double kernel_ms = std::chrono::duration<double, std::milli>(t_kernel_end - t_kernel_start).count();
    std::cout << "  Kernel execution: " << kernel_ms << " ms" << std::endl;

    // ---- Step 9: Read back results ----
    std::cout << "Reading back decoded data..." << std::endl;
    auto t_readback_start = std::chrono::high_resolution_clock::now();

    std::vector<uint32_t> result_u32;
    distributed::EnqueueReadMeshBuffer(cq, result_u32, out_dram_buf, /*blocking=*/true);

    auto t_readback_end = std::chrono::high_resolution_clock::now();
    double readback_ms = std::chrono::duration<double, std::milli>(t_readback_end - t_readback_start).count();
    std::cout << "  Readback time: " << readback_ms << " ms" << std::endl;

    // ---- Step 10: Verify bit-exactness ----
    std::cout << "\nVerifying bit-exactness..." << std::endl;
    const uint16_t* device_bf16 = reinterpret_cast<const uint16_t*>(result_u32.data());

    uint32_t mismatches = 0;
    uint32_t first_mismatch_idx = 0;
    for (uint32_t i = 0; i < num_elements; i++) {
        if (device_bf16[i] != cpu_bf16[i]) {
            if (mismatches == 0) {
                first_mismatch_idx = i;
            }
            mismatches++;
            if (mismatches <= 5) {
                std::cerr << "  MISMATCH at index " << i
                          << ": device=0x" << std::hex << device_bf16[i]
                          << " cpu=0x" << cpu_bf16[i] << std::dec << std::endl;
            }
        }
    }

    // ---- Step 11: Report metrics ----
    auto t_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    double compressed_mb = static_cast<double>(artifact.metadata.compressed_bytes) / (1024.0 * 1024.0);
    double output_mb = static_cast<double>(output_bytes) / (1024.0 * 1024.0);
    double decode_bw = output_mb / (kernel_ms / 1000.0);
    double cpu_bw = output_mb / (cpu_ms / 1000.0);

    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "  Elements decoded:    " << num_elements << std::endl;
    std::cout << "  Compressed size:     " << compressed_mb << " MB" << std::endl;
    std::cout << "  Output size:         " << output_mb << " MB" << std::endl;
    std::cout << "  Compression ratio:   " << artifact.metadata.compression_ratio << "x" << std::endl;
    std::cout << "  Bit-exact:           " << (mismatches == 0 ? "YES" : "NO") << std::endl;
    if (mismatches > 0) {
        std::cout << "  Mismatches:          " << mismatches << " / " << num_elements
                  << " (first at index " << first_mismatch_idx << ")" << std::endl;
    }
    std::cout << "\n=== Timing ===" << std::endl;
    std::cout << "  Artifact load:       " << load_ms << " ms" << std::endl;
    std::cout << "  LUT build:           " << lut_ms << " ms" << std::endl;
    std::cout << "  CPU decode:          " << cpu_ms << " ms (" << cpu_bw << " MB/s)" << std::endl;
    std::cout << "  Device init:         " << device_ms << " ms" << std::endl;
    std::cout << "  DRAM upload:         " << upload_ms << " ms" << std::endl;
    std::cout << "  Kernel execution:    " << kernel_ms << " ms (" << decode_bw << " MB/s)" << std::endl;
    std::cout << "  DRAM readback:       " << readback_ms << " ms" << std::endl;
    std::cout << "  Total wall clock:    " << total_ms << " ms" << std::endl;

    // Write metrics to JSON
    std::string metrics_path = artifact_dir + "/device_decode_metrics.json";
    {
        std::ofstream mf(metrics_path);
        mf << "{\n";
        mf << "  \"layer_name\": \"" << artifact.metadata.layer_name << "\",\n";
        mf << "  \"num_elements\": " << num_elements << ",\n";
        mf << "  \"compressed_bytes\": " << artifact.metadata.compressed_bytes << ",\n";
        mf << "  \"output_bytes\": " << output_bytes << ",\n";
        mf << "  \"compression_ratio\": " << artifact.metadata.compression_ratio << ",\n";
        mf << "  \"bit_exact\": " << (mismatches == 0 ? "true" : "false") << ",\n";
        mf << "  \"mismatches\": " << mismatches << ",\n";
        mf << "  \"artifact_load_ms\": " << load_ms << ",\n";
        mf << "  \"lut_build_ms\": " << lut_ms << ",\n";
        mf << "  \"cpu_decode_ms\": " << cpu_ms << ",\n";
        mf << "  \"cpu_bandwidth_mbs\": " << cpu_bw << ",\n";
        mf << "  \"device_init_ms\": " << device_ms << ",\n";
        mf << "  \"dram_upload_ms\": " << upload_ms << ",\n";
        mf << "  \"kernel_execution_ms\": " << kernel_ms << ",\n";
        mf << "  \"device_bandwidth_mbs\": " << decode_bw << ",\n";
        mf << "  \"dram_readback_ms\": " << readback_ms << ",\n";
        mf << "  \"total_ms\": " << total_ms << "\n";
        mf << "}\n";
        std::cout << "  Metrics saved to: " << metrics_path << std::endl;
    }

    // Optionally write decoded output for external comparison
    if (!output_file.empty()) {
        std::ofstream of(output_file, std::ios::binary);
        of.write(reinterpret_cast<const char*>(device_bf16), num_elements * 2);
        std::cout << "  Decoded BF16 written to: " << output_file << std::endl;
    }

    mesh_device->close();

    if (mismatches > 0) {
        std::cerr << "\nFAILED: " << mismatches << " mismatches detected" << std::endl;
        return 1;
    }

    std::cout << "\nSUCCESS: Device-side Huffman decode is bit-exact!" << std::endl;
    return 0;
}
