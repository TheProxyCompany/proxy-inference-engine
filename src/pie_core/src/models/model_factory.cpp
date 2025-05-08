#include "models/model_factory.hpp"
#include "models/model_config.hpp"
#include "models/model_registry.hpp"
#include "models/model_utils.hpp"
#include <spdlog/spdlog.h>
#include <spdlog/fmt/fmt.h>
#include <sstream>

#include <mlx/io.h>
#include <mlx/dtype.h>
#include <nlohmann/json.hpp>

// Add formatter for mlx::core::Dtype
template <> struct fmt::formatter<mlx::core::Dtype> {
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }
    template <typename FormatContext>
    auto format(const mlx::core::Dtype& dtype, FormatContext& ctx) const {
        std::stringstream ss;
        ss << dtype; // Use existing MLX stream operator
        return fmt::format_to(ctx.out(), "{}", ss.str());
    }
};

#include <string>
#include <unordered_map>
#include <memory>
#include <stdexcept>
#include <chrono>
#include <filesystem>

namespace pie_core::models {

    // Original function implementation delegates to the newer implementation
    std::unique_ptr<IModel> load_model(const std::string& model_path) {
        return load_model(model_path, std::nullopt);
    }

    // Implementation with optional EngineConfig
    std::unique_ptr<IModel> load_model(
        const std::string& model_path,
        const std::optional<engine::EngineConfig>& engine_config
    ) {
        spdlog::info("ModelFactory: Loading model from '{}' (EngineConfig provided: {})",
                    model_path, engine_config.has_value());
        auto start_time = std::chrono::steady_clock::now();

        // Verify model directory exists
        std::filesystem::path path(model_path);
        if (!std::filesystem::exists(path)) {
            spdlog::critical("ModelFactory: Model path '{}' does not exist", model_path);
            throw ModelLoadError("Model path '" + model_path + "' does not exist");
        }

        if (!std::filesystem::is_directory(path)) {
            spdlog::critical("ModelFactory: Model path '{}' is not a directory", model_path);
            throw ModelLoadError("Model path '" + model_path + "' is not a directory");
        }

        // 1. Determine model type
        spdlog::info("ModelFactory: Parsing base model configuration");
        ModelConfigBase base_config;
        try {
            base_config = parse_model_config_base(model_path);
            spdlog::info("ModelFactory: Detected model_type='{}'", base_config.model_type);
        } catch (const ConfigParseError& e) {
            spdlog::error("ModelFactory: Failed to parse base model configuration: {}", e.what());
            throw ModelLoadError("Failed to parse base config: " + std::string(e.what()));
        }

        // 2. Load all weights (common logic)
        spdlog::info("ModelFactory: Loading model weights");
        auto weights_start_time = std::chrono::steady_clock::now();
        std::unordered_map<std::string, mx::array> weights;

        try {
            spdlog::debug("ModelFactory: Calling load_all_weights to load model tensors");
            weights = load_all_weights(model_path);

            auto weights_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - weights_start_time).count();

            spdlog::info("ModelFactory: Successfully loaded {} weight tensors in {}ms",
                        weights.size(), weights_duration);

            // Log the first few weights at debug level
            int logged_count = 0;
            for (const auto& [key, tensor] : weights) {
                if (logged_count < 5) {  // Log up to 5 tensors for debugging
                    std::string shape_str = "[";
                    for (size_t i = 0; i < tensor.ndim(); i++) {
                        if (i > 0) shape_str += ", ";
                        shape_str += std::to_string(tensor.shape(i));
                    }
                    shape_str += "]";

                    spdlog::debug("ModelFactory: Weight tensor '{}' has shape {}",
                                 key, shape_str);
                    logged_count++;
                }
            }

        } catch (const std::exception& e) {
            spdlog::critical("ModelFactory: Failed to load model weights: {}", e.what());
            throw ModelLoadError("An unexpected error occurred during weight loading: " + std::string(e.what()));
        }

        // 3. Create model instance using the registry with optional EngineConfig
        spdlog::info("ModelFactory: Creating model instance of type '{}'", base_config.model_type);
        std::unique_ptr<IModel> model = nullptr;
        try {
            // Pass the engine_config to the registry
            model = ModelRegistry::create_model(base_config.model_type, model_path, engine_config);
            spdlog::debug("ModelFactory: Successfully created model instance of type '{}'", base_config.model_type);
        } catch (const std::runtime_error& e) {
            spdlog::critical("ModelFactory: Failed to create model instance of type '{}': {}",
                           base_config.model_type, e.what());
            throw ModelLoadError("Failed to create model instance: " + std::string(e.what()));
        }

        // 4. Load weights into the instantiated model
        spdlog::info("ModelFactory: Loading weights into model instance");
        auto load_weights_start = std::chrono::steady_clock::now();

        try {
            model->load_weights(weights);

            auto load_weights_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - load_weights_start).count();

            spdlog::info("ModelFactory: Successfully loaded weights into model instance in {}ms",
                        load_weights_duration);
        } catch (const std::runtime_error& e) {
            spdlog::critical("ModelFactory: Failed to load weights into model of type '{}': {}",
                           base_config.model_type, e.what());
            throw ModelLoadError("Failed to set weights for model type '" + base_config.model_type + "': " + e.what());
        }

        // Calculate and log total loading time
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start_time).count();

        // Log model parameters with attention type if available
        std::string attention_type_str = "";
        if (engine_config) {
            attention_type_str = fmt::format(", attention_type={}",
                engine_config->attention_type == engine::AttentionType::STANDARD ? "STANDARD" : "PAGED");
        }

        spdlog::info("ModelFactory: Model loaded successfully in {}ms - {} layers, {} KV heads, {} head dim, {} vocab size{}",
                    total_duration,
                    model->get_num_layers(),
                    model->get_num_kv_heads(),
                    model->get_head_dim(),
                    model->get_vocab_size(),
                    attention_type_str);

        return model;
    }

} // namespace pie_core::models
