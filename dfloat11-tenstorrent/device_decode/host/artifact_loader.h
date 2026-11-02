#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

struct ArtifactMetadata {
    std::string layer_name;
    std::vector<uint32_t> shape;
    std::string dtype;
    uint64_t original_bytes;
    uint64_t compressed_bytes;
    double compression_ratio;
    std::string checksum;
    uint32_t num_elements;
    uint64_t num_bits_exponent;
};

struct HuffmanCodeTable {
    std::unordered_map<uint32_t, std::string> code_table;  // symbol -> bitstring
};

struct CompressedArtifact {
    ArtifactMetadata metadata;
    HuffmanCodeTable codec;
    std::vector<uint8_t> exponent_bitstream;
    std::vector<uint8_t> sign_mantissa;
};

namespace {

std::string read_file_to_string(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    return std::string((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
}

std::vector<uint8_t> read_binary_file(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs.is_open()) {
        throw std::runtime_error("Cannot open binary file: " + path);
    }
    auto size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    std::vector<uint8_t> data(size);
    ifs.read(reinterpret_cast<char*>(data.data()), size);
    return data;
}

// Minimal JSON string value parser - extracts value for a given key from simple JSON
std::string json_get_string(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) throw std::runtime_error("Key not found: " + key);
    pos = json.find(':', pos);
    auto start = json.find('"', pos + 1);
    auto end_pos = json.find('"', start + 1);
    return json.substr(start + 1, end_pos - start - 1);
}

uint64_t json_get_uint64(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) throw std::runtime_error("Key not found: " + key);
    pos = json.find(':', pos);
    auto start = pos + 1;
    while (start < json.size() && (json[start] == ' ' || json[start] == '\n')) start++;
    auto end_pos = start;
    while (end_pos < json.size() && (json[end_pos] >= '0' && json[end_pos] <= '9')) end_pos++;
    return std::stoull(json.substr(start, end_pos - start));
}

double json_get_double(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) throw std::runtime_error("Key not found: " + key);
    pos = json.find(':', pos);
    auto start = pos + 1;
    while (start < json.size() && (json[start] == ' ' || json[start] == '\n')) start++;
    auto end_pos = start;
    while (end_pos < json.size() && (json[end_pos] == '.' || json[end_pos] == '-' ||
           (json[end_pos] >= '0' && json[end_pos] <= '9'))) end_pos++;
    return std::stod(json.substr(start, end_pos - start));
}

std::vector<uint32_t> json_get_int_array(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) throw std::runtime_error("Key not found: " + key);
    auto bracket_start = json.find('[', pos);
    auto bracket_end = json.find(']', bracket_start);
    std::string arr_str = json.substr(bracket_start + 1, bracket_end - bracket_start - 1);
    std::vector<uint32_t> result;
    std::istringstream ss(arr_str);
    std::string token;
    while (std::getline(ss, token, ',')) {
        auto trimmed = token;
        while (!trimmed.empty() && (trimmed.front() == ' ' || trimmed.front() == '\n')) trimmed.erase(trimmed.begin());
        while (!trimmed.empty() && (trimmed.back() == ' ' || trimmed.back() == '\n')) trimmed.pop_back();
        if (!trimmed.empty()) {
            result.push_back(static_cast<uint32_t>(std::stoul(trimmed)));
        }
    }
    return result;
}

// Parse codec.json: {"code_table": {"symbol": "bitstring", ...}, ...}
HuffmanCodeTable parse_codec_json(const std::string& json) {
    HuffmanCodeTable table;
    std::string ct_key = "\"code_table\"";
    auto pos = json.find(ct_key);
    if (pos == std::string::npos) throw std::runtime_error("code_table not found in codec.json");
    auto brace_start = json.find('{', pos + ct_key.size());
    int depth = 1;
    auto i = brace_start + 1;
    while (i < json.size() && depth > 0) {
        if (json[i] == '{') depth++;
        else if (json[i] == '}') depth--;
        if (depth > 0) i++;
    }
    std::string ct_str = json.substr(brace_start + 1, i - brace_start - 1);

    // Parse key-value pairs: "symbol": "bitstring"
    size_t scan = 0;
    while (scan < ct_str.size()) {
        auto key_start = ct_str.find('"', scan);
        if (key_start == std::string::npos) break;
        auto key_end = ct_str.find('"', key_start + 1);
        std::string sym_str = ct_str.substr(key_start + 1, key_end - key_start - 1);

        auto val_start = ct_str.find('"', key_end + 1);
        auto val_end = ct_str.find('"', val_start + 1);
        std::string bits_str = ct_str.substr(val_start + 1, val_end - val_start - 1);

        uint32_t symbol = static_cast<uint32_t>(std::stoul(sym_str));
        table.code_table[symbol] = bits_str;
        scan = val_end + 1;
    }
    return table;
}

// Read NPY file (uint8 only, C-contiguous, 1D)
std::vector<uint8_t> read_npy_uint8(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) throw std::runtime_error("Cannot open npy: " + path);

    // NPY magic: \x93NUMPY
    char magic[6];
    ifs.read(magic, 6);
    if (magic[0] != '\x93' || std::string(magic + 1, 5) != "NUMPY") {
        throw std::runtime_error("Not a valid .npy file: " + path);
    }

    uint8_t major, minor;
    ifs.read(reinterpret_cast<char*>(&major), 1);
    ifs.read(reinterpret_cast<char*>(&minor), 1);

    uint32_t header_len = 0;
    if (major == 1) {
        uint16_t hl;
        ifs.read(reinterpret_cast<char*>(&hl), 2);
        header_len = hl;
    } else {
        ifs.read(reinterpret_cast<char*>(&header_len), 4);
    }

    std::string header(header_len, '\0');
    ifs.read(&header[0], header_len);

    // Extract shape from header, e.g. "'shape': (262668288,)"
    auto shape_pos = header.find("'shape'");
    if (shape_pos == std::string::npos) shape_pos = header.find("\"shape\"");
    auto paren_start = header.find('(', shape_pos);
    auto paren_end = header.find(')', paren_start);
    std::string shape_str = header.substr(paren_start + 1, paren_end - paren_start - 1);
    // Remove trailing comma
    while (!shape_str.empty() && (shape_str.back() == ',' || shape_str.back() == ' ')) shape_str.pop_back();
    uint64_t total_elements = 0;
    if (!shape_str.empty()) {
        total_elements = std::stoull(shape_str);
    }

    std::vector<uint8_t> data(total_elements);
    ifs.read(reinterpret_cast<char*>(data.data()), total_elements);
    return data;
}

}  // anonymous namespace

inline CompressedArtifact load_compressed_artifact(const std::string& artifact_dir) {
    CompressedArtifact artifact;

    std::string meta_json = read_file_to_string(artifact_dir + "/metadata.json");
    artifact.metadata.layer_name = json_get_string(meta_json, "layer_name");
    artifact.metadata.shape = json_get_int_array(meta_json, "shape");
    artifact.metadata.dtype = json_get_string(meta_json, "dtype");
    artifact.metadata.original_bytes = json_get_uint64(meta_json, "original_bytes");
    artifact.metadata.compressed_bytes = json_get_uint64(meta_json, "compressed_bytes");
    artifact.metadata.compression_ratio = json_get_double(meta_json, "compression_ratio");
    artifact.metadata.checksum = json_get_string(meta_json, "checksum");
    artifact.metadata.num_elements = static_cast<uint32_t>(json_get_uint64(meta_json, "num_elements"));
    artifact.metadata.num_bits_exponent = json_get_uint64(meta_json, "num_bits_exponent");

    std::string codec_json = read_file_to_string(artifact_dir + "/codec.json");
    artifact.codec = parse_codec_json(codec_json);

    artifact.exponent_bitstream = read_binary_file(artifact_dir + "/exponent_bitstream.bin");
    artifact.sign_mantissa = read_npy_uint8(artifact_dir + "/sign_mantissa.npy");

    return artifact;
}
