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
#include "attention/AttentionRegistry.hpp" // For auto-registration

// Placeholder kernel source (replace with actual loading if preferred later)
const std::string PAGED_ATTENTION_KERNEL_SOURCE = R"(
#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

// Basic placeholder kernel
[[kernel]]
void paged_attention_kernel(
    const device float* queries         [[buffer(0)]], // Input name "queries"
    device float*       output          [[buffer(1)]], // Output name "output"
    constant const int& some_param      [[buffer(2)]], // Example parameter
    uint tid [[thread_position_in_grid]] // Thread ID
    ) {

    // Simple dummy operation: copy first element of query based on thread ID
    // This is just to make the kernel compile and run.
    // Replace with actual paged attention logic later.
    if (tid < 1) { // Avoid out-of-bounds in this stub
         output[tid] = queries[0] + float(some_param);
    } else {
         output[tid] = 0.0f;
    }
}
)";

namespace pie_core::attention {

mx::array PagedAttentionMechanism::compute(
    const mx::array& queries,
    const mx::array& keys,
    const mx::array& values,
    const engine::BatchDetails& details
) const {
    spdlog::trace("PagedAttentionMechanism: Preparing to invoke custom Metal kernel.");

    if (queries.size() == 0) {
        spdlog::warn("PagedAttentionMechanism: Queries array is empty, returning empty array.");
        return mx::array({}, queries.dtype());
    }
    if (queries.dtype() != mx::float32) {
         // Placeholder kernel expects float32
         spdlog::error("PagedAttentionMechanism: Placeholder kernel expects float32 queries, got {}.", queries.dtype());
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
        std::vector<std::pair<std::string, mx::Dtype>> template_args = {}; // None for placeholder

        // Define grid and threadgroup sizes. Adjust based on kernel logic.
        // For the placeholder that only accesses tid=0, a small grid is fine.
        // For real attention, grid should cover all query tokens/heads.
        size_t grid_dim_x = std::max(1UL, queries.size()); // Example: One thread per query element (adjust later)
        MTL::Size grid = MTL::Size(grid_dim_x, 1, 1);
        MTL::Size threadgroup = MTL::Size(std::min(256UL, grid_dim_x), 1, 1); // Common threadgroup size

        // Define output shapes and types
        std::vector<std::vector<int>> output_shapes = {queries.shape()}; // Output has same shape as query
        std::vector<mx::Dtype> output_dtypes = {queries.dtype()};

        // Example constant buffer data (needs to match kernel's constant args)
        // For the placeholder kernel expecting `constant const int& some_param [[buffer(2)]]`
        std::unordered_map<std::string, int> constant_data = {{"some_param", 42}};

        spdlog::trace("PagedAttentionMechanism: Invoking Metal kernel 'paged_attention'...");
        // 3. Call the kernel
        auto outputs = kernel(
            kernel_inputs,
            template_args,
            constant_data, // Pass constant data map
            grid,
            threadgroup,
            output_shapes,
            output_dtypes
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
