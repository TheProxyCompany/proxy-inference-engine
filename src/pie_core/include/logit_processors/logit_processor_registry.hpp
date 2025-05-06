#pragma once

#include "logit_processors/logit_processor.hpp"
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <stdexcept>

namespace pie_core::logit_processors {
    using LogitProcessorCreatorFunc = std::function<std::unique_ptr<ILogitProcessor>()>;

    class LogitProcessorRegistry {
    public:
        static bool register_processor(const std::string& processor_type, LogitProcessorCreatorFunc creator);

        static std::unique_ptr<ILogitProcessor> create_processor(const std::string& processor_type);

        LogitProcessorRegistry(const LogitProcessorRegistry&) = delete;
        LogitProcessorRegistry& operator=(const LogitProcessorRegistry&) = delete;

    private:
        LogitProcessorRegistry() = default;
        static std::unordered_map<std::string, LogitProcessorCreatorFunc>& get_registry() {
            static std::unordered_map<std::string, LogitProcessorCreatorFunc> registry;
            return registry;
        }
    };

    template <typename T>
    class LogitProcessorRegistrar {
    public:
        LogitProcessorRegistrar(const std::string& processor_type) {
            LogitProcessorRegistry::register_processor(processor_type, []() {
                return std::make_unique<T>();
            });
        }
    };
}
