#pragma once

#include "attention/IAttentionMechanism.hpp" // Include the interface
#include <functional>                      // For std::function
#include <memory>                          // For std::unique_ptr
#include <string>                          // For std::string
#include <unordered_map>                   // For std::unordered_map
#include <stdexcept>                       // For std::runtime_error
#include "engine/batch_details.hpp"        // For AttentionType enum

namespace pie_core::attention {

// Alias for the factory function type
using AttentionMechanismCreatorFunc = std::function<std::unique_ptr<IAttentionMechanism>()>;

/**
 * @brief Registry for different attention mechanism implementations.
 *
 * Allows registering and creating attention mechanisms based on their type.
 * Uses the Singleton pattern (via static methods) for global access.
 */
class AttentionRegistry {
public:
    /**
     * @brief Registers a new attention mechanism type.
     * @param type The type identifier (e.g., AttentionType enum value).
     * @param creator A function that creates an instance of the mechanism.
     * @return True if registration was successful, false otherwise (e.g., type already registered).
     * @throws std::runtime_error if the type is already registered.
     */
    static bool register_mechanism(engine::AttentionType type, AttentionMechanismCreatorFunc creator);

    /**
     * @brief Creates an instance of the specified attention mechanism type.
     * @param type The type of mechanism to create.
     * @return A unique_ptr to the created attention mechanism instance.
     * @throws std::runtime_error if the type is not registered.
     */
    static std::unique_ptr<IAttentionMechanism> create_mechanism(engine::AttentionType type);

    // Delete copy/move constructors and assignment operators for the registry itself
    AttentionRegistry(const AttentionRegistry&) = delete;
    AttentionRegistry& operator=(const AttentionRegistry&) = delete;
    AttentionRegistry(AttentionRegistry&&) = delete;
    AttentionRegistry& operator=(AttentionRegistry&&) = delete;

private:
    // Private constructor to prevent instantiation
    AttentionRegistry() = default;

    // Static function to get the underlying map (singleton instance)
    static std::unordered_map<engine::AttentionType, AttentionMechanismCreatorFunc>& get_registry();
};

/**
 * @brief Helper class to automatically register attention mechanisms upon construction.
 *
 * Usage: Create a static instance of this class within the .cpp file of the
 *        attention mechanism implementation you want to register.
 *
 * Example (in StandardAttentionMechanism.cpp):
 * namespace {
 *     AttentionMechanismRegistrar<StandardAttentionMechanism> registrar(engine::AttentionType::STANDARD);
 * }
 *
 * @tparam T The concrete attention mechanism class to register.
 */
template <typename T>
class AttentionMechanismRegistrar {
public:
    explicit AttentionMechanismRegistrar(engine::AttentionType type) {
        // Register a lambda that creates a new instance of the mechanism T
        AttentionRegistry::register_mechanism(type, []() {
            return std::make_unique<T>();
        });
    }
};

} // namespace pie_core::attention
