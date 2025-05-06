#pragma once

#include <mlx/mlx.h>
#include <cstdint>
#include <memory>
#include <vector>

namespace mx = mlx::core;

namespace pie_core::models {

    struct BatchDetails;

    class IModel {
    public:
        virtual ~IModel() = default;

        // --- Core Inference Method ---
        virtual mx::array forward(const engine::BatchDetails& batch_details) const = 0;

        // --- Parameter Management ---
        virtual std::vector<mx::array*> get_parameters() = 0;
        virtual void load_weights(const std::unordered_map<std::string, mx::array>& weights) = 0;

        // --- Structural Information for Scheduler/Allocator ---
        virtual int get_num_kv_heads() const noexcept = 0;
        virtual int get_head_dim() const noexcept = 0;
        virtual int get_num_layers() const noexcept = 0;
        virtual size_t get_vocab_size() const noexcept = 0;

        virtual bool supports_multimodal() const noexcept { return false; }
        virtual std::vector<std::string> supported_modalities() const noexcept { return {"text"}; }
    };

} // namespace pie_core
