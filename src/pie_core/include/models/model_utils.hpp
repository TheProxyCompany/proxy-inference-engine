#pragma once

#include <mlx/array.h>
#include <string>
#include <unordered_map>
#include <filesystem>

namespace mx = mlx::core;

namespace pie_core::models {

    namespace fs = std::filesystem;

    std::optional<fs::path> find_gguf_file(const fs::path& model_path);

    std::unordered_map<std::string, mx::array>
    load_sharded_safetensors_weights(const fs::path& model_path, const fs::path& index_path);

    std::unordered_map<std::string, mx::array>
    load_single_safetensors_weights(const fs::path& single_file_path);

    std::unordered_map<std::string, mx::array>
    load_gguf_weights(const fs::path& gguf_file_path);

    std::unordered_map<std::string, mx::array>
    load_all_weights(const std::string& model_path_str);


} // namespace pie_core::models
