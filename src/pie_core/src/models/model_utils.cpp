#include "models/model_factory.hpp"

#include <mlx/io.h>
#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <set>
#include <memory>
#include <stdexcept>
#include <iostream>

#include "models/model_utils.hpp"

namespace mx = mlx::core;

namespace pie_core::models {

    std::optional<fs::path> find_gguf_file(const fs::path& model_path) {
            std::optional<fs::path> found_path = std::nullopt;
            int count = 0;
            for (const auto& entry : fs::directory_iterator(model_path)) {
                if (entry.is_regular_file() && entry.path().extension() == ".gguf") {
                    if (count == 0) {
                        found_path = entry.path();
                    }
                    count++;
                }
            }
            if (count > 1 && found_path.has_value()) {
                std::cerr << "Warning: Found " << count << " '.gguf' files in " << model_path << ". Using the first one found: " << found_path.value().filename() << std::endl;
            }
            return found_path;
        }

        std::unordered_map<std::string, mx::array>
        load_sharded_safetensors_weights(const fs::path& model_path, const fs::path& index_path) {
            std::ifstream index_stream(index_path);
            if (!index_stream.is_open()) {
                throw ModelLoadError("Failed to open weight index file: " + index_path.string());
            }

            nlohmann::json index_json;
            try {
                index_json = nlohmann::json::parse(index_stream);
            } catch (const nlohmann::json::parse_error& e) {
                throw ModelLoadError("Failed to parse weight index JSON: " + std::string(e.what()));
            }

            if (!index_json.contains("weight_map") || !index_json["weight_map"].is_object()) {
                 throw ModelLoadError("Invalid weight index JSON format: missing or invalid 'weight_map'");
            }

            std::set<std::string> shard_files_to_load;
            for (const auto& item : index_json["weight_map"].items()) {
                 if (!item.value().is_string()) {
                    continue;
                 }
                shard_files_to_load.insert(item.value().get<std::string>());
            }

            if (shard_files_to_load.empty()) {
                 throw ModelLoadError("Weight index file contains no valid shard references.");
            }

            std::unordered_map<std::string, mx::array> all_weights;
            for (const auto& shard_filename : shard_files_to_load) {
                fs::path shard_path = model_path / shard_filename;
                if (!fs::exists(shard_path)) {
                    throw ModelLoadError("Weight shard file not found: " + shard_path.string());
                }
                try {
                    auto shard_data = mx::load_safetensors(shard_path.string());
                    for (auto&& [key, val] : shard_data.first) {
                         all_weights.try_emplace(key, std::move(val));
                    }
                } catch (const std::exception& e) {
                     throw ModelLoadError("Failed to load weight shard '" + shard_filename + "': " + e.what());
                }
            }
            std::cout << "Finished loading shards." << std::endl;
            return all_weights;
        }

        std::unordered_map<std::string, mx::array>
        load_single_safetensors_weights(const fs::path& single_file_path) {
            try {
                auto loaded_data = mx::load_safetensors(single_file_path.string());
                return loaded_data.first;
            } catch (const std::exception& e) {
                throw ModelLoadError("Failed to load single weight file '" + single_file_path.string() + "': " + e.what());
            }
        }

        std::unordered_map<std::string, mx::array>
        load_gguf_weights(const fs::path& gguf_file_path) {
             try {
                auto loaded_data = mx::load_gguf(gguf_file_path.string());
                return loaded_data.first;
             } catch (const std::exception& e) {
                 throw ModelLoadError("Failed to load GGUF weight file '" + gguf_file_path.string() + "': " + e.what());
             }
        }

        std::unordered_map<std::string, mx::array> load_all_weights(const std::string& model_path_str) {
            fs::path model_path = model_path_str;
            fs::path index_path = model_path / "model.safetensors.index.json";
            fs::path single_safetensors_path = model_path / "model.safetensors";
            std::optional<fs::path> gguf_path_opt = find_gguf_file(model_path);

            std::unordered_map<std::string, mx::array> all_weights;

            if (fs::exists(index_path)) {
                all_weights = load_sharded_safetensors_weights(model_path, index_path);
            } else if (fs::exists(single_safetensors_path)) {
                all_weights = load_single_safetensors_weights(single_safetensors_path);
            } else if (gguf_path_opt.has_value()) {
                all_weights = load_gguf_weights(gguf_path_opt.value());
            } else {
                throw ModelLoadError("No weights found in: " + model_path_str);
            }

            if (all_weights.empty()) {
                throw ModelLoadError("Loaded weights map is empty from: " + model_path_str);
            }

            std::cout << "Successfully loaded " << all_weights.size() << " weight tensors." << std::endl;
            return all_weights;
        }

} // namespace pie_core::models
