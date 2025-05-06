#include "models/llama3/llama3_config.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>

namespace pie_core::models::llama3 {

    LlamaConfig parse_llama_config(const std::string& model_dir_path) {
        namespace fs = std::filesystem;
        fs::path config_file_path = fs::path(model_dir_path) / "config.json";

        std::ifstream config_stream(config_file_path);
        if (!config_stream.is_open()) {
            throw ConfigParseError("Failed to open Llama config file: " + config_file_path.string());
        }

        nlohmann::json config_json;
        try {
            config_json = nlohmann::json::parse(config_stream);
        } catch (const nlohmann::json::parse_error& e) {
            throw ConfigParseError("Failed to parse Llama config JSON: " + std::string(e.what()));
        }

        LlamaConfig config;

        try {
            config.model_type = config_json.value("model_type", config.model_type);
            config.hidden_size = config_json.value("hidden_size", config.hidden_size);
            config.num_hidden_layers = config_json.value("num_hidden_layers", config.num_hidden_layers);
            config.intermediate_size = config_json.value("intermediate_size", config.intermediate_size);
            config.num_attention_heads = config_json.value("num_attention_heads", config.num_attention_heads);
            config.num_key_value_heads = config_json.value("num_key_value_heads", config.num_key_value_heads);
            config.rms_norm_eps = config_json.value("rms_norm_eps", config.rms_norm_eps);
            config.vocab_size = config_json.value("vocab_size", config.vocab_size);
            config.max_position_embeddings = config_json.value("max_position_embeddings", config.max_position_embeddings);
            config.rope_theta = config_json.value("rope_theta", config.rope_theta);
            config.rope_traditional = config_json.value("rope_traditional", config.rope_traditional);
            config.attention_bias = config_json.value("attention_bias", config.attention_bias);
            config.mlp_bias = config_json.value("mlp_bias", config.mlp_bias);
            config.tie_word_embeddings = config_json.value("tie_word_embeddings", config.tie_word_embeddings);

            // Parse optional rope_scaling dictionary
            if (config_json.contains("rope_scaling") && config_json["rope_scaling"].is_object()) {
                const auto& rope_scaling_json = config_json["rope_scaling"];
                Llama3RopeConfig rope_config;

                // Populate base RoPEConfig fields first
                rope_config.dims = config.hidden_size / config.num_attention_heads;
                rope_config.traditional = config.rope_traditional;
                rope_config.base = config.rope_theta;
                rope_config.scale = rope_scaling_json.value("factor", 1.0f);

                // Populate Llama3 specific RoPE fields
                rope_config.max_position_embeddings = config.max_position_embeddings;
                rope_config.original_max_position_embeddings = rope_scaling_json.value(
                    "original_max_position_embeddings", config.max_position_embeddings);
                rope_config.factor = rope_scaling_json.value("factor", 1.0f);
                rope_config.low_freq_factor = rope_scaling_json.value("low_freq_factor", 1.0f);
                rope_config.high_freq_factor = rope_scaling_json.value("high_freq_factor", 1.0f);

                config.rope_scaling = rope_config;
            }

        } catch (const nlohmann::json::exception& e) {
            throw ConfigParseError("Error parsing Llama config fields: " + std::string(e.what()));
        }

        if (config.model_type != "llama") {
            throw ConfigParseError("Expected model_type 'llama' but found '" + config.model_type + "'");
        }

        return config;
    }

} // namespace pie_core::models::llama3
