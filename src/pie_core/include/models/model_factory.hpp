#pragma once

#include "models/imodel.hpp"
#include <string>
#include <memory>
#include <stdexcept>
#include <optional>

namespace pie_core::engine {
    struct EngineConfig; // Forward declaration
}

namespace pie_core::models {

    class ModelLoadError : public std::runtime_error {
    public:
        using std::runtime_error::runtime_error;
    };

    // Original function (kept for backward compatibility)
    std::unique_ptr<IModel> load_model(const std::string& model_path);

    // New function that accepts optional EngineConfig
    std::unique_ptr<IModel> load_model(
        const std::string& model_path,
        const std::optional<engine::EngineConfig>& engine_config
    );

} // namespace pie_core::models
