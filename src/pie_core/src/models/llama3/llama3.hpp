#pragma once

#include "engine/batch_details.hpp"
#include "models/imodel.hpp"
#include "models/llama3/llama3_config.hpp"
#include "layers/embedding.hpp"
#include "layers/transformer_block.hpp"
#include "layers/norm.hpp"
#include "layers/linear.hpp"
#include <vector>
#include <memory>

namespace pie_core::models::llama3 {

    class LlamaModel : public IModel {
    public:
        explicit LlamaModel(const LlamaConfig& config);

        mx::array forward(const engine::BatchDetails& batch_details) override;

        int get_num_kv_heads() const noexcept override;
        int get_head_dim() const noexcept override;
        int get_num_layers() const noexcept override;
        size_t get_vocab_size() const noexcept override;

        std::vector<mx::array*> get_parameters() override;
        void load_weights(const std::unordered_map<std::string, mx::array>& weights) override;

    private:
        LlamaConfig config_;
        layers::Embedding embed_tokens_;
        std::vector<layers::TransformerBlock> layers_;
        layers::RMSNorm norm_;
        std::optional<layers::Linear> lm_head_;
    };

} // namespace pie_core::models::llama3
