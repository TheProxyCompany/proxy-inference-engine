#include "models/model_registry.hpp"
#include "engine/engine.hpp" // Include full definition for EngineConfig
#include <spdlog/spdlog.h>

namespace pie_core::models {

    // Registry singleton implementation
    std::unordered_map<std::string, ModelCreatorFunc>& ModelRegistry::get_registry() {
        static std::unordered_map<std::string, ModelCreatorFunc> registry;
        return registry;
    }

    bool ModelRegistry::register_model(const std::string& model_type, ModelCreatorFunc creator) {
        auto& registry = get_registry();
        if (registry.count(model_type)) {
            spdlog::error("Model type already registered: {}", model_type);
            throw std::runtime_error("Model type already registered: " + model_type);
        }
        registry[model_type] = std::move(creator);
        return true;
    }

    std::unique_ptr<IModel> ModelRegistry::create_model(
        const std::string& model_type,
        const std::string& model_path,
        const std::optional<engine::EngineConfig>& engine_config
    ) {
        auto& registry = get_registry();
        auto it = registry.find(model_type);
        if (it == registry.end()) {
            spdlog::error("Unsupported model type requested: {}", model_type);
            throw std::runtime_error("Unsupported model type: " + model_type);
        }

        spdlog::debug("Creating model of type '{}' using path '{}' (EngineConfig provided: {})",
                     model_type, model_path, engine_config.has_value());

        // Call the registered factory function, passing the path and the optional config
        return it->second(model_path, engine_config);
    }

} // namespace pie_core::models
