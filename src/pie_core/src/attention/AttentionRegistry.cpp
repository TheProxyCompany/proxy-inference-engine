#include "attention/AttentionRegistry.hpp"
#include "engine/batch_details.hpp" // For AttentionType enum
#include <stdexcept>               // For runtime_error
#include <spdlog/spdlog.h>         // For logging

namespace pie_core::attention {

// Definition for the static function to get the registry map
std::unordered_map<engine::AttentionType, AttentionMechanismCreatorFunc>&
AttentionRegistry::get_registry() {
    // Static local variable ensures it's initialized only once (thread-safe in C++11+)
    static std::unordered_map<engine::AttentionType, AttentionMechanismCreatorFunc> registry_instance;
    return registry_instance;
}

bool AttentionRegistry::register_mechanism(engine::AttentionType type, AttentionMechanismCreatorFunc creator) {
    auto& registry = get_registry();
    if (registry.count(type)) {
        // Log an error, but maybe don't throw in production? Or throw in debug?
        // Throwing is safer to catch configuration errors early.
        spdlog::error("Attention mechanism type '{}' already registered.", static_cast<int>(type));
        throw std::runtime_error("Attention mechanism type already registered.");
        // return false; // Alternative: return false if throwing is too harsh
    }
    registry[type] = std::move(creator);
    spdlog::debug("Registered attention mechanism type '{}'.", static_cast<int>(type));
    return true;
}

std::unique_ptr<IAttentionMechanism> AttentionRegistry::create_mechanism(engine::AttentionType type) {
    auto& registry = get_registry();
    auto it = registry.find(type);
    if (it == registry.end()) {
        spdlog::error("Unsupported attention mechanism type requested: '{}'.", static_cast<int>(type));
        throw std::runtime_error("Unsupported attention mechanism type requested.");
    }
    spdlog::debug("Creating attention mechanism of type '{}'.", static_cast<int>(type));
    // Call the registered factory function
    return it->second();
}

} // namespace pie_core::attention
