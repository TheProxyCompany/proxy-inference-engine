#include "attention/PagedAttentionMechanism.hpp"
#include "engine/batch_details.hpp" // Include BatchDetails definition
#include <mlx/ops.h>               // For basic ops
#include <mlx/fast.h>              // For metal_kernel
#include <Metal/Metal.hpp>         // For MTL::Size
#include <spdlog/spdlog.h>         // For logging
#include <stdexcept>               // For runtime_error
#include <vector>                  // For input/output lists
#include <string>                  // For source string
#include <unordered_map>           // For constant_data map
#include <sstream>                 // For std::stringstream
#include <tuple>                   // For std::tuple
#include <optional>                // For std::optional and std::nullopt
#include "attention/AttentionRegistry.hpp" // For auto-registration

// Placeholder kernel source (Simplified)
const std::string PAGED_ATTENTION_KERNEL_SOURCE = R"(
#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

// Basic placeholder kernel (Simplified - no constant buffer)
[[kernel]]
void paged_attention_kernel(
    const device float* queries         [[buffer(0)]], // Input name "queries"
    device float*       output          [[buffer(1)]], // Output name "output"
    // constant const int& some_param      [[buffer(2)]], // REMOVED FOR NOW
    uint tid [[thread_position_in_grid]] // Thread ID
    ) {

    // Simple dummy operation (Removed dependency on some_param)
    if (tid < 1) {
         output[tid] = queries[0]; // Just copy first element
    } else {
         output[tid] = 0.0f;
    }
}
)";

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
    if (queries.dtype() != mx::float32) {
         // Placeholder kernel expects float32
         std::stringstream ss;
         ss << queries.dtype(); // Use stream operator
         spdlog::error("PagedAttentionMechanism: Placeholder kernel expects float32 queries, got {}.", ss.str());
         throw std::runtime_error("Placeholder kernel requires float32 queries.");
    }

    try {
        // 1. Define the kernel using mx::fast::metal_kernel
        auto kernel = mx::fast::metal_kernel(
            /* name= */ "paged_attention", // User-defined name for caching/debugging
            /* input_names= */ {"queries"}, // Matches kernel arg name (excluding buffer index)
            /* output_names= */ {"output"}, // Matches kernel arg name (excluding buffer index)
            /* source= */ PAGED_ATTENTION_KERNEL_SOURCE
            // Optional: Add template_names if kernel uses templates
        );

        // 2. Prepare inputs for the kernel call
        std::vector<mx::array> kernel_inputs = {queries};

        // Define template args (if any). Placeholder kernel doesn't use templates effectively yet.
        std::vector<std::pair<std::string, mlx::core::fast::TemplateArg>> template_args = {}; // None for placeholder

        // Define grid and threadgroup sizes. Adjust based on kernel logic.
        // For the placeholder that only accesses tid=0, a small grid is fine.
        // For real attention, grid should cover all query tokens/heads.
        size_t grid_dim_x = std::max(1UL, queries.size()); // Example: One thread per query element (adjust later)
        // Use std::tuple<int, int, int> instead of MTL::Size
        std::tuple<int, int, int> grid = {(int)grid_dim_x, 1, 1};
        std::tuple<int, int, int> threadgroup = {(int)std::min(256UL, grid_dim_x), 1, 1}; // Common threadgroup size

        // Define output shapes and types
        std::vector<std::vector<int>> output_shapes = {queries.shape()}; // Output has same shape as query
        std::vector<mx::Dtype> output_dtypes = {queries.dtype()};

        // Constant buffer data removed to simplify (since we removed it from the kernel)
        // std::unordered_map<std::string, int> constant_data = {{"some_param", 42}};

        spdlog::trace("PagedAttentionMechanism: Invoking Metal kernel 'paged_attention'...");
        // 3. Call the kernel - Removed constant_data argument
        auto outputs = kernel(
            kernel_inputs,     // const std::vector<array>& inputs
            output_shapes,     // const std::vector<std::vector<int>>& output_shapes
            output_dtypes,     // const std::vector<Dtype>& output_dtypes
            grid,              // const std::tuple<int, int, int>& grid_dims
            threadgroup,       // const std::tuple<int, int, int>& block_dims
            template_args,     // const std::vector<std::pair<std::string, ...>>& template_args
            std::nullopt,      // The missing optional<float> scale argument
            false,             // bool ensure_row_contiguous = false
            {}                 // StreamOrDevice stream = {}
        );
        spdlog::trace("PagedAttentionMechanism: Metal kernel execution completed.");

        // 4. Return the output tensor
        if (outputs.empty()) {
             throw std::runtime_error("Metal kernel returned no outputs.");
        }
        return outputs[0];

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
