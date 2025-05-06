#include "logit_processors/logit_processor_registry.hpp"

namespace pie_core::logit_processors {

    bool LogitProcessorRegistry::register_processor(const std::string& processor_type, LogitProcessorCreatorFunc creator) {
        auto& registry = get_registry();
        if (registry.count(processor_type)) {
            throw std::runtime_error("Logit processor type already registered: " + processor_type);
        }
        registry[processor_type] = std::move(creator);
        return true;
    }

    std::unique_ptr<ILogitProcessor> LogitProcessorRegistry::create_processor(const std::string& processor_type) {
        auto& registry = get_registry();
        auto it = registry.find(processor_type);
        if (it == registry.end()) {
            throw std::runtime_error("Unsupported logit processor type: " + processor_type);
        }
        return it->second();
    }

}
