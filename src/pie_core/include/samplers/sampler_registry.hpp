#pragma once
#include "samplers/isampler.hpp"
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <stdexcept>

namespace pie_core::samplers {
    using SamplerCreatorFunc = std::function<std::unique_ptr<ISampler>()>;

    class SamplerRegistry {
    public:
        static bool register_sampler(const std::string& sampler_type, SamplerCreatorFunc creator);

        static std::unique_ptr<ISampler> create_sampler(const std::string& sampler_type);

        SamplerRegistry(const SamplerRegistry&) = delete;
        SamplerRegistry& operator=(const SamplerRegistry&) = delete;

    private:
        SamplerRegistry() = default;
        static std::unordered_map<std::string, SamplerCreatorFunc>& get_registry() {
            static std::unordered_map<std::string, SamplerCreatorFunc> registry;
            return registry;
        }
    };

    template <typename T>
    class SamplerRegistrar {
    public:
        SamplerRegistrar(const std::string& sampler_type) {
            SamplerRegistry::register_sampler(sampler_type, []() {
                return std::make_unique<T>();
            });
        }
    };
}
