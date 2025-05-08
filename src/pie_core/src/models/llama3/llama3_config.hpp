#pragma once

#include "layers/rope.hpp"
#include "models/model_config.hpp"
#include "engine/batch_details.hpp" // For AttentionType
#include <string>
#include <optional>

namespace pie_core::models::llama3 {

namespace engine = pie_core::engine; // Alias for engine namespace

    /**
     * @brief Configuration specific to the RoPE variant used in Llama 3.
     *
     * This struct extends the base RoPE configuration with parameters
     * necessary for the nuanced frequency calculations particular to Llama 3,
     * as observed in reference implementations. It anticipates a potential
     * C++ RoPE layer implementation that mirrors the frequency precomputation
     * logic found in Python counterparts.
     */
    struct Llama3RopeConfig : public layers::RoPEConfig {
        int max_position_embeddings;
        int original_max_position_embeddings;
        float factor;
        float low_freq_factor;
        float high_freq_factor;
    };

    struct LlamaConfig : public ModelConfigBase {
        std::string model_type = "llama";
        int hidden_size = 4096;
        int num_hidden_layers = 32;
        int intermediate_size = 14336;
        int num_attention_heads = 32;
        int num_key_value_heads = 8;
        float rms_norm_eps = 1e-5f;
        int vocab_size = 128256;
        int max_position_embeddings = 8192;
        float rope_theta = 500000.0f;
        bool rope_traditional = false;
        std::optional<Llama3RopeConfig> rope_scaling = std::nullopt;
        bool attention_bias = false;
        bool mlp_bias = false;
        bool tie_word_embeddings = false;
        engine::AttentionType attention_type = engine::AttentionType::STANDARD; // Default to standard attention

        Llama3RopeConfig get_rope_config() const {
            return rope_scaling.value_or(Llama3RopeConfig{
                .max_position_embeddings = max_position_embeddings,
                .original_max_position_embeddings = max_position_embeddings,
                .factor = 1.0f,
                .low_freq_factor = 1.0f,
                .high_freq_factor = 1.0f
            });
        }
    };

    LlamaConfig parse_llama_config(const std::string& config_path);

} // namespace pie_core::models::llama
