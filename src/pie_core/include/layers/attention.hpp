#pragma once

#include "layers/linear.hpp"
#include "layers/rope.hpp"
#include "engine/batch_details.hpp"        // Keep BatchDetails include
#include "attention/IAttentionMechanism.hpp" // Include the mechanism interface
#include "attention/AttentionRegistry.hpp" // Include the registry
#include <mlx/mlx.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <optional>
#include <memory> // For std::unique_ptr

namespace pie_core::layers {

namespace mx = mlx::core;
namespace attention = pie_core::attention; // Alias for brevity
namespace engine = pie_core::engine;       // Alias for brevity

/**
 * @brief Configuration struct specific to the Attention layer.
 */
struct AttentionConfig {
    int hidden_dims;
    int num_heads;
    int num_kv_heads;
    RoPEConfig rope_config;
    bool bias = false;
    engine::AttentionType attention_type = engine::AttentionType::STANDARD; // Add attention type
};

/**
 * @brief Implements Multi-Head (or Grouped-Query) Attention using a pluggable mechanism.
 */
class Attention {
public:
    /**
     * @brief Constructs the Attention layer and selects the attention mechanism.
     * @param config Configuration parameters, including the desired attention_type.
     */
    explicit Attention(const AttentionConfig& config);

    // Rule of 5/6
    Attention(const Attention&) = delete;
    Attention& operator=(const Attention&) = delete;
    Attention(Attention&&) = default; // Default move is fine with unique_ptr
    Attention& operator=(Attention&&) = default; // Default move is fine with unique_ptr
    ~Attention() = default; // Default destructor is fine with unique_ptr

    /**
     * @brief Performs the attention forward pass by delegating to the selected mechanism.
     * @param hidden_state Input tensor from the previous layer.
     * @param batch_details Contains consolidated block tables, sequence info, etc.
     * @return Output tensor after attention calculation and output projection.
     */
    mx::array forward(
        const mx::array& hidden_state,
        const engine::BatchDetails& batch_details
    ) const;

    mx::array operator()(
        const mx::array& hidden_state,
        const engine::BatchDetails& batch_details
    ) const {
        return forward(hidden_state, batch_details);
    }

    /**
     * @brief Delegates loading weights to its internal Linear layers.
     * @param weights Map containing all model weights.
     * @param prefix Prefix for keys belonging to this Attention block (e.g., "self_attn.").
     */
    void load_weights(const std::unordered_map<std::string, mx::array>& weights,
                      const std::string& prefix);

    /**
     * @brief Delegates parameter collection to its internal Linear layers.
     * @param params Vector to which parameter pointers will be added.
     */
    void collect_parameters(std::vector<mx::array*>& params);

private:
    AttentionConfig config_; // Store the config

    // --- Sub-layers ---
    Linear q_proj_;
    Linear k_proj_;
    Linear v_proj_;
    Linear o_proj_;
    RoPE rope_;

    // --- Selected Attention Mechanism ---
    std::unique_ptr<attention::IAttentionMechanism> mechanism_;

};

} // namespace pie_core::layers
