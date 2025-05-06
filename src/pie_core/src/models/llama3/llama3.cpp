#include "models/llama3/llama3.hpp"
#include "engine/batch_details.hpp"
#include "models/model_registry.hpp"
#include <stdexcept>

namespace pie_core::models::llama3 {

    LlamaModel::LlamaModel(const LlamaConfig& config)
        : config_(config),
          embed_tokens_(config.vocab_size, config.hidden_size),
          norm_(config.hidden_size, config.rms_norm_eps)
    {
        layers_.reserve(config.num_hidden_layers);
        for (int i = 0; i < config.num_hidden_layers; ++i) {
            layers::RoPEConfig rope_config = config.get_rope_config();
            layers::AttentionConfig attn_config = {
                .hidden_dims = config.hidden_size,
                .num_heads = config.num_attention_heads,
                .num_kv_heads = config.num_key_value_heads,
                .rope_config = rope_config,
                .bias = config.attention_bias
            };
            layers::TransformerBlockConfig block_config = {
                .hidden_dims = config.hidden_size,
                .mlp_hidden_dims = config.intermediate_size,
                .norm_eps = config.rms_norm_eps,
                .attn_config = attn_config
            };
            layers_.emplace_back(block_config);
        }

        if (!config.tie_word_embeddings) {
            lm_head_.emplace(config.hidden_size, config.vocab_size, /*bias=*/false);
        }
    }

    mx::array LlamaModel::forward(const engine::BatchDetails& batch_details) const {
        // 1. Get embeddings from token IDs in batch_details
        mx::array hidden_state = embed_tokens_.forward(batch_details.token_ids);

        // 2. Pass through Transformer blocks
        for (const auto& layer : layers_) {
            hidden_state = layer.forward(hidden_state, batch_details);
        }

        // 3. Final normalization
        hidden_state = norm_.forward(hidden_state);

        // 4. Language model head projection
        if (lm_head_.has_value()) {
            return lm_head_->forward(hidden_state);
        } else {
            return embed_tokens_.as_linear(hidden_state);
        }
    }

    std::vector<mx::array*> LlamaModel::get_parameters() {
        std::vector<mx::array*> params;
        embed_tokens_.collect_parameters(params);
        for (auto& layer : layers_) {
            layer.collect_parameters(params);
        }
        norm_.collect_parameters(params);
        if (lm_head_.has_value()) {
            lm_head_->collect_parameters(params);
        }
        return params;
    }

    // --- Getters for IModel Interface ---
    int LlamaModel::get_num_kv_heads() const noexcept {
        return config_.num_key_value_heads;
    }

    int LlamaModel::get_head_dim() const noexcept {
        if (config_.num_attention_heads == 0) return 0;
        return config_.hidden_size / config_.num_attention_heads;
    }

    int LlamaModel::get_num_layers() const noexcept {
        return config_.num_hidden_layers;
    }

    size_t LlamaModel::get_vocab_size() const noexcept {
        return static_cast<size_t>(config_.vocab_size);
    }
    // --- End Getters ---


    // --- Weight Loading (Implementation) ---
    void LlamaModel::load_weights(const std::unordered_map<std::string, mx::array>& weights) {
         try {
             embed_tokens_.load_weights(weights, "model.embed_tokens.");
             for (int i = 0; i < config_.num_hidden_layers; ++i) {
                 layers_[i].load_weights(weights, "model.layers." + std::to_string(i) + ".");
             }
             norm_.load_weights(weights, "model.norm.");
             if (lm_head_.has_value()) {
                 lm_head_->load_weights(weights, "lm_head.");
             }
         } catch (const std::runtime_error& e) {
              throw std::runtime_error("Failed to load weights for LlamaModel: " + std::string(e.what()));
         }
    }

    namespace {
        std::unique_ptr<IModel> create_llama_model(const std::string& model_path) {
            LlamaConfig config = parse_llama_config(model_path);
            return std::make_unique<LlamaModel>(config);
        }

        struct LlamaRegistrar {
            LlamaRegistrar() {
                ModelRegistry::register_model("llama", create_llama_model);
            }
        };
        static LlamaRegistrar registrar_instance;
    }

} // namespace pie_core::models::llama3
