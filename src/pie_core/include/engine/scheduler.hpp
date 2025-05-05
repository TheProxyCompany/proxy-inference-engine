#pragma once

#include <memory>
#include <cstdint>
#include <optional>

namespace mlx::core { class Module; }
namespace pie_core {
    class PageAllocator;
    class Sequence;
    class IModel;
}

namespace pie_core::engine {

    /**
     * @brief Orchestrates LLM inference requests, managing batching and resources.
     */
    class Scheduler {
    public:
        /**
         * @brief Constructor. Initializes the scheduler with necessary components.
         * @param allocator A reference to the PageAllocator for KV cache management.
         * @param model A unique pointer to the loaded model object (Scheduler takes ownership).
         * @param max_num_seqs Max concurrent sequences the scheduler will manage.
         * @param max_tokens_in_batch Max total tokens per GPU batch.
         */
        Scheduler(
            PageAllocator& allocator,
            std::unique_ptr<IModel> model,
            size_t max_num_seqs = 256,
            size_t max_tokens_in_batch = 4096
        );

        /**
         * @brief Destructor. Required for PImpl to work correctly.
         */
        ~Scheduler();

        /**
         * @brief Enqueues a new sequence request for processing.
         * @param sequence Unique pointer to the sequence object. Scheduler takes ownership if accepted.
         * @return True if the request was accepted, false otherwise (e.g., queue full).
         */
        bool add_request(std::unique_ptr<Sequence> sequence);

        /**
         * @brief Executes a single step of the scheduler's main loop.
         * @return True if any work was performed (batch executed), false if idle.
         */
        bool step();

        // --- Prevent Copying/Moving ---
        Scheduler(const Scheduler&) = delete;
        Scheduler& operator=(const Scheduler&) = delete;
        Scheduler(Scheduler&&) = delete;
        Scheduler& operator=(Scheduler&&) = delete;

    private:
        // --- PImpl (Pointer to Implementation) ---
        // Forward declare the implementation struct/class.
        struct SchedulerImpl;

        // The unique pointer holding the actual implementation details.
        std::unique_ptr<SchedulerImpl> pimpl_;
    };

} // namespace pie_core
