#include "models/model_factory.hpp"
#include <spdlog/spdlog.h>

#include <mlx/io.h>
#include <mlx/dtype.h>
#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <set>
#include <memory>
#include <stdexcept>
#include <chrono>
#include <functional>

// Add hash specialization for mlx::core::Dtype
namespace std {
    template <>
    struct hash<mlx::core::Dtype> {
        std::size_t operator()(const mlx::core::Dtype& k) const {
            // Hash the underlying enum value
            return std::hash<int>()(static_cast<int>(k.val()));
        }
    };
} // namespace std

#include "models/model_utils.hpp"

namespace mx = mlx::core;

namespace pie_core::models {

    std::optional<fs::path> find_gguf_file(const fs::path& model_path) {
        spdlog::debug("ModelUtils: Searching for .gguf files in '{}'", model_path.string());

        std::optional<fs::path> found_path = std::nullopt;
        int count = 0;
        std::vector<fs::path> all_gguf_files;

        for (const auto& entry : fs::directory_iterator(model_path)) {
            if (entry.is_regular_file() && entry.path().extension() == ".gguf") {
                if (count == 0) {
                    found_path = entry.path();
                }
                all_gguf_files.push_back(entry.path());
                count++;
            }
        }

        if (count == 0) {
            spdlog::debug("ModelUtils: No .gguf files found in '{}'", model_path.string());
            return std::nullopt;
        } else if (count == 1) {
            spdlog::debug("ModelUtils: Found 1 .gguf file: '{}'", found_path.value().string());
            return found_path;
        } else {
            // Multiple files found, log all of them
            std::string files_list;
            for (size_t i = 0; i < all_gguf_files.size(); i++) {
                if (i > 0) files_list += ", ";
                files_list += "'" + all_gguf_files[i].filename().string() + "'";
            }

            spdlog::warn("ModelUtils: Found {} .gguf files in '{}': {}. Using the first one: '{}'",
                        count, model_path.string(), files_list, found_path.value().filename().string());
            return found_path;
        }
    }

    std::unordered_map<std::string, mx::array>
    load_sharded_safetensors_weights(const fs::path& model_path, const fs::path& index_path) {
        spdlog::info("ModelUtils: Loading sharded safetensors weights using index file '{}'", index_path.string());
        auto start_time = std::chrono::steady_clock::now();

        // Open index file
        std::ifstream index_stream(index_path);
        if (!index_stream.is_open()) {
            spdlog::error("ModelUtils: Failed to open weight index file: '{}'", index_path.string());
            throw ModelLoadError("Failed to open weight index file: " + index_path.string());
        }

        // Parse JSON
        nlohmann::json index_json;
        try {
            index_json = nlohmann::json::parse(index_stream);
            spdlog::debug("ModelUtils: Successfully parsed index JSON");
        } catch (const nlohmann::json::parse_error& e) {
            spdlog::error("ModelUtils: JSON parse error in index file '{}': {}", index_path.string(), e.what());
            throw ModelLoadError("Failed to parse weight index JSON: " + std::string(e.what()));
        }

        // Validate JSON structure
        if (!index_json.contains("weight_map") || !index_json["weight_map"].is_object()) {
            spdlog::error("ModelUtils: Invalid index JSON format: missing or invalid 'weight_map' field");
            throw ModelLoadError("Invalid weight index JSON format: missing or invalid 'weight_map'");
        }

        // Extract shard filenames
        std::set<std::string> shard_files_to_load;
        for (const auto& item : index_json["weight_map"].items()) {
            if (!item.value().is_string()) {
                spdlog::warn("ModelUtils: Skipping non-string value for weight key '{}' in index", item.key());
                continue;
            }
            shard_files_to_load.insert(item.value().get<std::string>());
        }

        if (shard_files_to_load.empty()) {
            spdlog::error("ModelUtils: No valid shard references found in index file '{}'", index_path.string());
            throw ModelLoadError("Weight index file contains no valid shard references.");
        }

        spdlog::info("ModelUtils: Found {} shard files to load", shard_files_to_load.size());

        // Load each shard
        std::unordered_map<std::string, mx::array> all_weights;
        size_t shard_count = 0;
        size_t total_tensors = 0;

        for (const auto& shard_filename : shard_files_to_load) {
            shard_count++;
            fs::path shard_path = model_path / shard_filename;

            if (!fs::exists(shard_path)) {
                spdlog::error("ModelUtils: Weight shard file not found: '{}'", shard_path.string());
                throw ModelLoadError("Weight shard file not found: " + shard_path.string());
            }

            spdlog::info("ModelUtils: Loading shard {}/{}: '{}'",
                        shard_count, shard_files_to_load.size(), shard_filename);

            try {
                auto shard_load_start = std::chrono::steady_clock::now();

                auto shard_data = mx::load_safetensors(shard_path.string());
                size_t shard_tensor_count = shard_data.first.size();
                total_tensors += shard_tensor_count;

                auto shard_load_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - shard_load_start).count();

                spdlog::debug("ModelUtils: Loaded shard '{}' with {} tensors in {}ms",
                            shard_filename, shard_tensor_count, shard_load_time);

                // Move tensors to main weight map
                for (auto&& [key, val] : shard_data.first) {
                    all_weights.try_emplace(key, std::move(val));
                }

            } catch (const std::exception& e) {
                spdlog::error("ModelUtils: Failed to load weight shard '{}': {}", shard_filename, e.what());
                throw ModelLoadError("Failed to load weight shard '" + shard_filename + "': " + e.what());
            }
        }

        auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start_time).count();

        spdlog::info("ModelUtils: Successfully loaded {} tensors from {} shards in {}ms",
                    total_tensors, shard_files_to_load.size(), total_time);

        return all_weights;
    }

    std::unordered_map<std::string, mx::array>
    load_single_safetensors_weights(const fs::path& single_file_path) {
        spdlog::info("ModelUtils: Loading single safetensors file: '{}'", single_file_path.string());
        auto start_time = std::chrono::steady_clock::now();

        try {
            auto loaded_data = mx::load_safetensors(single_file_path.string());
            size_t tensor_count = loaded_data.first.size();

            auto load_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start_time).count();

            spdlog::info("ModelUtils: Successfully loaded {} tensors from '{}' in {}ms",
                        tensor_count, single_file_path.filename().string(), load_time);

            return loaded_data.first;
        } catch (const std::exception& e) {
            spdlog::error("ModelUtils: Failed to load single safetensors file '{}': {}",
                         single_file_path.string(), e.what());
            throw ModelLoadError("Failed to load single weight file '" + single_file_path.string() + "': " + e.what());
        }
    }

    std::unordered_map<std::string, mx::array>
    load_gguf_weights(const fs::path& gguf_file_path) {
        spdlog::info("ModelUtils: Loading GGUF file: '{}'", gguf_file_path.string());
        auto start_time = std::chrono::steady_clock::now();

        try {
            auto loaded_data = mx::load_gguf(gguf_file_path.string());
            size_t tensor_count = loaded_data.first.size();

            auto load_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start_time).count();

            spdlog::info("ModelUtils: Successfully loaded {} tensors from GGUF file '{}' in {}ms",
                        tensor_count, gguf_file_path.filename().string(), load_time);

            // Log metadata if available
            if (!loaded_data.second.empty()) {
                spdlog::debug("ModelUtils: GGUF file contains {} metadata entries", loaded_data.second.size());

                // Log a few key metadata entries if they exist
                const std::vector<std::string> key_metadata = {
                    "general.architecture", "general.name", "llama.context_length",
                    "tokenizer.ggml.model", "general.quantization_version"
                };

                for (const auto& key : key_metadata) {
                    if (loaded_data.second.count(key) > 0) {
                        const auto& value = loaded_data.second.at(key);
                        if (std::holds_alternative<std::string>(value)) {
                            spdlog::debug("ModelUtils: GGUF metadata '{}' = '{}'",
                                        key, std::get<std::string>(value));
                        }
                    }
                }
            }

            return loaded_data.first;
        } catch (const std::exception& e) {
            spdlog::error("ModelUtils: Failed to load GGUF file '{}': {}",
                        gguf_file_path.string(), e.what());
            throw ModelLoadError("Failed to load GGUF weight file '" + gguf_file_path.string() + "': " + e.what());
        }
    }

    std::unordered_map<std::string, mx::array> load_all_weights(const std::string& model_path_str) {
        spdlog::info("ModelUtils: Loading model weights from '{}'", model_path_str);
        auto start_time = std::chrono::steady_clock::now();

        fs::path model_path = model_path_str;

        // Check potential weight files
        fs::path index_path = model_path / "model.safetensors.index.json";
        fs::path single_safetensors_path = model_path / "model.safetensors";

        spdlog::debug("ModelUtils: Checking for weight files:");

        bool has_index = fs::exists(index_path);
        spdlog::debug("ModelUtils: - Sharded safetensors index (model.safetensors.index.json): {}",
                     has_index ? "FOUND" : "NOT FOUND");

        bool has_single_safetensors = fs::exists(single_safetensors_path);
        spdlog::debug("ModelUtils: - Single safetensors file (model.safetensors): {}",
                     has_single_safetensors ? "FOUND" : "NOT FOUND");

        std::optional<fs::path> gguf_path_opt = find_gguf_file(model_path);
        bool has_gguf = gguf_path_opt.has_value();
        spdlog::debug("ModelUtils: - GGUF file (*.gguf): {}",
                     has_gguf ? "FOUND" : "NOT FOUND");

        std::unordered_map<std::string, mx::array> all_weights;

        if (has_index) {
            spdlog::info("ModelUtils: Loading weights from sharded safetensors files");
            all_weights = load_sharded_safetensors_weights(model_path, index_path);
        } else if (has_single_safetensors) {
            spdlog::info("ModelUtils: Loading weights from single safetensors file");
            all_weights = load_single_safetensors_weights(single_safetensors_path);
        } else if (has_gguf) {
            spdlog::info("ModelUtils: Loading weights from GGUF file");
            all_weights = load_gguf_weights(gguf_path_opt.value());
        } else {
            spdlog::critical("ModelUtils: No weight files found in directory '{}'", model_path_str);
            throw ModelLoadError("No weights found in: " + model_path_str);
        }

        if (all_weights.empty()) {
            spdlog::critical("ModelUtils: Loaded weights map is empty from '{}'", model_path_str);
            throw ModelLoadError("Loaded weights map is empty from: " + model_path_str);
        }

        auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start_time).count();

        spdlog::info("ModelUtils: Successfully loaded {} weight tensors in {}ms", all_weights.size(), total_time);

        // Log tensor stats
        size_t total_params = 0;
        std::unordered_map<mx::Dtype, size_t> dtype_counts;

        for (const auto& [name, tensor] : all_weights) {
            size_t tensor_size = 1;
            for (size_t i = 0; i < tensor.ndim(); i++) {
                tensor_size *= tensor.shape(i);
            }
            total_params += tensor_size;
            dtype_counts[tensor.dtype()]++;
        }

        // Log dtype distribution
        for (const auto& [dtype, count] : dtype_counts) {
            spdlog::debug("ModelUtils: {} tensors", count);
        }

        spdlog::debug("ModelUtils: Total parameter count: {}", total_params);

        return all_weights;
    }

} // namespace pie_core::models
