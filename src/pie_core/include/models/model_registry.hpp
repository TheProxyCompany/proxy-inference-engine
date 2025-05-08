#pragma once

#include "models/imodel.hpp"
#include "engine/engine.hpp" // Include full definition
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <stdexcept>
#include <optional>

namespace pie_core::models {

    // Update signature to accept optional EngineConfig
    using ModelCreatorFunc = std::function<std::unique_ptr<IModel>(
        const std::string&,
        const std::optional<engine::EngineConfig>&
    )>;

    class ModelRegistry {
    public:
        // Registration signature remains the same
        static bool register_model(const std::string& model_type, ModelCreatorFunc creator);

        // Update signature to accept optional EngineConfig
        static std::unique_ptr<IModel> create_model(
            const std::string& model_type,
            const std::string& model_path,
            const std::optional<engine::EngineConfig>& engine_config = std::nullopt
        );

        // Delete copy/move constructors and assignment operators
        ModelRegistry(const ModelRegistry&) = delete;
        ModelRegistry& operator=(const ModelRegistry&) = delete;
        ModelRegistry(ModelRegistry&&) = delete;
        ModelRegistry& operator=(ModelRegistry&&) = delete;

    private:
        ModelRegistry() = default;

        // Static function to get the underlying map (singleton instance)
        static std::unordered_map<std::string, ModelCreatorFunc>& get_registry();
    };

    // Update the registrar template to handle the optional config
    template <typename T>
    class ModelRegistrar {
    public:
        ModelRegistrar(const std::string& model_type) {
            ModelRegistry::register_model(model_type,
                // Lambda now accepts optional engine_config
                [](const std::string& model_path, const std::optional<engine::EngineConfig>& engine_config) -> std::unique_ptr<IModel> {
                    // Parse config from model files
                    auto config = T::parse_config(model_path);

                    // Apply runtime config if available
                    if (engine_config) {
                        // Assuming T has a method to update its config from EngineConfig
                        // This will be implemented in specific model classes like Llama3
                        if constexpr (requires { T::apply_engine_config(config, *engine_config); }) {
                            T::apply_engine_config(config, *engine_config);
                        }
                    }

                    // Construct model with potentially modified config
                    return std::make_unique<T>(config);
                }
            );
        }
    };

} // namespace pie_core::models
