#include "samplers/sampler_registry.hpp"

namespace pie_core::samplers {

    bool SamplerRegistry::register_sampler(const std::string& sampler_type, SamplerCreatorFunc creator) {
        auto& registry = get_registry();
        if (registry.count(sampler_type)) {
            throw std::runtime_error("Sampler type already registered: " + sampler_type);
        }
        registry[sampler_type] = std::move(creator);
        return true;
    }

    std::unique_ptr<ISampler> SamplerRegistry::create_sampler(const std::string& sampler_type) {
        auto& registry = get_registry();
        auto it = registry.find(sampler_type);
        if (it == registry.end()) {
            throw std::runtime_error("Unsupported sampler type: " + sampler_type);
        }
        return it->second();
    }


}
