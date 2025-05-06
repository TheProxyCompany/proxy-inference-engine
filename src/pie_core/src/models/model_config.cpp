#include "models/model_config.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>

namespace pie_core::models {

    ModelConfigBase parse_model_config_base(const std::string& model_dir_path) {
        namespace fs = std::filesystem;
        fs::path config_file_path = fs::path(model_dir_path) / "config.json";

        std::ifstream config_stream(config_file_path);
        if (!config_stream.is_open()) {
            throw ConfigParseError("Failed to open config file: " + config_file_path.string());
        }

        nlohmann::json config_json;
        try {
            config_json = nlohmann::json::parse(config_stream);
        } catch (const nlohmann::json::parse_error& e) {
            throw ConfigParseError("Failed to parse config JSON: " + std::string(e.what()));
        }

        ModelConfigBase base_config;
        try {
            if (!config_json.contains("model_type") || !config_json["model_type"].is_string()) {
                 throw ConfigParseError("Missing or invalid 'model_type' in config.json");
            }
            base_config.model_type = config_json.at("model_type").get<std::string>();
        } catch (const nlohmann::json::exception& e) {
             throw ConfigParseError("Error accessing 'model_type' in config JSON: " + std::string(e.what()));
        }

        return base_config;
    }

} // namespace pie_core::models
