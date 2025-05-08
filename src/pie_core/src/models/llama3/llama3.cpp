#include "models/llama3/llama3.hpp"
#include "engine/batch_details.hpp"
#include "models/model_registry.hpp"
#include "engine/engine.hpp" // For EngineConfig
#include <mlx/ops.h> // For triu, full
#include <limits>   // For infinity
#include <spdlog/spdlog.h>
#include <spdlog/fmt/ranges.h> // For formatting vectors like shape
#include <stdexcept>
#include <optional> // For std::optional

namespace pie_core::models::llama3 {

    LlamaModel::LlamaModel(const LlamaConfig& config)
        : config_(config),
          embed_tokens_(config.vocab_size, config.hidden_size),
          norm_(config.hidden_size, config.rms_norm_eps)
    {
        spdlog::info("LlamaModel: Constructing with AttentionType: {}",
                    static_cast<int>(config.attention_type));
        layers_.reserve(config.num_hidden_layers);
        for (int i = 0; i < config.num_hidden_layers; ++i) {
            layers::RoPEConfig rope_config = config.get_rope_config();
            layers::AttentionConfig attn_config = {
                .hidden_dims = config.hidden_size,
                .num_heads = config.num_attention_heads,
                .num_kv_heads = config.num_key_value_heads,
                .rope_config = rope_config,
                .bias = config.attention_bias,
                .attention_type = config.attention_type // Pass the attention type from LlamaConfig
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

        std::optional<mx::array> attention_mask = std::nullopt;

        // 2. Create Causal Mask if using standard attention
        if (batch_details.attention_type == engine::AttentionType::STANDARD) {
            // This mask creation assumes hidden_state is [B, L, D].
            if (hidden_state.ndim() == 3) {
                int L = hidden_state.shape(1); // Sequence length
                mx::array mask = mx::triu(mx::full({L, L}, -std::numeric_limits<float>::infinity(), hidden_state.dtype()), /*k=*/1);
                attention_mask = mask;
                spdlog::trace("LlamaModel::forward: Created causal mask for standard attention (L={})", L);
            } else if (hidden_state.ndim() == 2 && batch_details.sequence_ids.size() == 1) {
                // Handle flattened input for a single sequence batch [TotalTokens, D]
                int L = hidden_state.shape(0);
                mx::array mask = mx::triu(mx::full({L, L}, -std::numeric_limits<float>::infinity(), hidden_state.dtype()), /*k=*/1);
                attention_mask = mask;
                spdlog::trace("LlamaModel::forward: Created causal mask for standard attention (single sequence, L={})", L);
            } else {
                // If the shape isn't [B, L, D] or [L, D] for single seq, we can't easily make a standard causal mask.
                spdlog::warn("LlamaModel::forward: Cannot create standard causal mask for hidden_state shape {}. "
                             "Expected [B, L, D] or [L, D] for AttentionType::STANDARD.",
                             fmt::format("{}", hidden_state.shape())); // Format shape vector
                attention_mask = std::nullopt;
            }
        } else {
            spdlog::trace("LlamaModel::forward: Skipping mask creation for paged attention.");
            // For PAGED attention, masking is handled internally by the kernel based on context lengths.
        }

        // 3. Pass through Transformer blocks
        for (const auto& layer : layers_) {
            hidden_state = layer.forward(hidden_state, batch_details);
        }

        // 4. Final normalization
        hidden_state = norm_.forward(hidden_state);

        // 5. Language model head projection
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

    // The apply_engine_config method has been removed as it's not needed here
    // The lambda in the registry directly applies the configuration

    // --- Model Registry Integration ---
    namespace { // Use anonymous namespace

        // The lambda passed to the registrar now matches the updated ModelCreatorFunc signature
        std::unique_ptr<IModel> create_llama_model_registered(
            const std::string& model_path,
            const std::optional<engine::EngineConfig>& engine_config // Receive optional config
        ) {
            // 1. Parse the configuration from the model directory
            LlamaConfig llama_config = parse_llama_config(model_path);
            spdlog::debug("Llama Creator: Parsed base config for '{}'. Default AttentionType: {}",
                          model_path, static_cast<int>(llama_config.attention_type));

            // 2. Override config fields based on engine_config if provided
            if (engine_config) {
                spdlog::debug("Llama Creator: Applying AttentionType ({}) from EngineConfig.",
                              static_cast<int>(engine_config.value().attention_type));
                llama_config.attention_type = engine_config.value().attention_type;
                // Add overrides for other relevant EngineConfig fields here if needed in the future
            } else {
                 spdlog::debug("Llama Creator: No EngineConfig provided, using default AttentionType from parsed config.");
            }

            // 3. Construct the LlamaModel with the finalized configuration
            spdlog::info("Llama Creator: Constructing LlamaModel with final AttentionType: {}",
                         static_cast<int>(llama_config.attention_type));
            return std::make_unique<LlamaModel>(llama_config);
        }

        struct LlamaRegistrar {
            LlamaRegistrar() {
                // Register the updated creator function
                ModelRegistry::register_model("llama", create_llama_model_registered);
                spdlog::debug("LlamaModel registered with ModelRegistry.");
            }
        };
        static LlamaRegistrar registrar_instance; // Static instance ensures registration

    } // anonymous namespace

} // namespace pie_core::models::llama3
