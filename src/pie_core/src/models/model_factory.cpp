#include "models/model_factory.hpp"
#include "models/model_config.hpp"
#include "models/model_registry.hpp"
#include "models/model_utils.hpp"

#include <mlx/io.h>
#include <nlohmann/json.hpp>

#include <string>
#include <unordered_map>
#include <memory>
#include <stdexcept>
#include <iostream>

namespace pie_core::models {

    std::unique_ptr<IModel> load_model(const std::string& model_path) {
        // 1. Determine model type
        ModelConfigBase base_config;
        try {
            base_config = parse_model_config_base(model_path);
        } catch (const ConfigParseError& e) {
            throw ModelLoadError("Failed to parse base config: " + std::string(e.what()));
        }

        // 2. Load all weights (common logic)
        std::unordered_map<std::string, mx::array> weights;
        try {
            weights = load_all_weights(model_path);
        } catch (const std::exception& e) {
            throw ModelLoadError("An unexpected error occurred during weight loading: " + std::string(e.what()));
        }

        // 3. Create model instance using the registry
        std::unique_ptr<IModel> model = nullptr;
        try {
            model = ModelRegistry::create_model(base_config.model_type, model_path);
        } catch (const std::runtime_error& e) {
             throw ModelLoadError("Failed to create model instance: " + std::string(e.what()));
        }

        // 4. Load weights into the instantiated model
        try {
            model->load_weights(weights);
        } catch (const std::runtime_error& e) {
            throw ModelLoadError("Failed to set weights for model type '" + base_config.model_type + "': " + e.what());
        }

        return model;
    }

} // namespace pie_core::models
