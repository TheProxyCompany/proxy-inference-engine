#include "models/model_registry.hpp"

namespace pie_core::models {

    bool ModelRegistry::register_model(const std::string& model_type, ModelCreatorFunc creator) {
        auto& registry = get_registry();
        if (registry.count(model_type)) {
            throw std::runtime_error("Model type already registered: " + model_type);
        }
        registry[model_type] = std::move(creator);
        return true;
    }

    std::unique_ptr<IModel> ModelRegistry::create_model(const std::string& model_type, const std::string& model_path) {
        auto& registry = get_registry();
        auto it = registry.find(model_type);
        if (it == registry.end()) {
            throw std::runtime_error("Unsupported model type: " + model_type);
        }
        return it->second(model_path);
    }

} // namespace pie_core::models
