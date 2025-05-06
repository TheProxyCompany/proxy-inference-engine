#pragma once

#include "models/imodel.hpp"
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <stdexcept>

namespace pie_core::models {

    using ModelCreatorFunc = std::function<std::unique_ptr<IModel>(const std::string&)>;

    class ModelRegistry {
    public:
        static bool register_model(const std::string& model_type, ModelCreatorFunc creator);

        static std::unique_ptr<IModel> create_model(const std::string& model_type, const std::string& model_path);

        ModelRegistry(const ModelRegistry&) = delete;
        ModelRegistry& operator=(const ModelRegistry&) = delete;

    private:
        ModelRegistry() = default;

        static std::unordered_map<std::string, ModelCreatorFunc>& get_registry() {
            static std::unordered_map<std::string, ModelCreatorFunc> registry;
            return registry;
        }
    };

    template <typename T>
    class ModelRegistrar {
    public:
        ModelRegistrar(const std::string& model_type) {
            ModelRegistry::register_model(model_type,
                [](const std::string& model_path) -> std::unique_ptr<IModel> {
                    auto config = T::parse_config(model_path);
                    return std::make_unique<T>(config);
                }
            );
        }
    };

} // namespace pie_core::models
