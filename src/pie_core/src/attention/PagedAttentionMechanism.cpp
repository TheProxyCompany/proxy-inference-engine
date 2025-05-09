#include "attention/PagedAttentionMechanism.hpp"
#include "engine/batch_details.hpp" // Include BatchDetails definition
#include <mlx/ops.h>               // For basic ops
#include <Metal/Metal.hpp>         // For MTL::Size
#include <mlx/device.h>            // For mlx::Device
#include <spdlog/spdlog.h>         // For logging
#include <stdexcept>               // For runtime_error
#include <vector>                  // For input/output lists
#include <string>                  // For source string
#include <unordered_map>           // For constant_data map
#include <sstream>                 // For std::stringstream
#include <tuple>                   // For std::tuple
#include <optional>                // For std::optional and std::nullopt
#include "attention/AttentionRegistry.hpp" // For auto-registration

#include "ops.hpp"

namespace pie_core::attention {

mx::array PagedAttentionMechanism::compute(
    const mx::array& queries,
    const mx::array& keys [[maybe_unused]],
    const mx::array& values [[maybe_unused]],
    const engine::BatchDetails& details [[maybe_unused]]
) const {
    spdlog::trace("PagedAttentionMechanism: Preparing to invoke custom Metal kernel.");

    if (queries.size() == 0) {
        spdlog::warn("PagedAttentionMechanism: Queries array is empty, returning empty array.");
        return mx::array({}, queries.dtype());
    }

    try {
        mx::array mock_kv_cache_for_pal = mx::ones_like(queries); // Placeholder
        mx::array mock_page_table_for_pal = mx::zeros({queries.shape(0)}, mx::uint32); // Placeholder
        auto stream = mx::default_stream(mx::default_device());

        spdlog::trace("PagedAttentionMechanism: Invoking PAL's paged_attention C++ operation.");
        mx::array output = pal::cpp::paged_attention(
            queries,
            mock_kv_cache_for_pal,
            mock_page_table_for_pal,
            stream
        );

        spdlog::trace("PagedAttentionMechanism: PAL's paged_attention operation completed.");
        return output;

    } catch (const std::exception& e) {
        spdlog::error("PagedAttentionMechanism: Error during custom Metal kernel invocation: {}", e.what());
        throw; // Re-throw the exception
    }
}

// --- Auto-registration ---
namespace { // Use anonymous namespace
    AttentionMechanismRegistrar<PagedAttentionMechanism> registrar(engine::AttentionType::PAGED);
} // anonymous namespace

} // namespace pie_core::attention
