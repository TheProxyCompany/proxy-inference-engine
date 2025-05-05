#pragma once

#include <mlx/mlx.h>
#include <vector>
#include <cstdint>
#include <string>
#include <memory>
#include <atomic>
#include <limits>
#include <optional>

#include "sequence/sampling_params.hpp"
#include "sequence/logits_params.hpp"
#include "sequence/stop_criteria.hpp"
#include "sequence/ipc_handles.hpp"

namespace mx = mlx::core;

namespace pie_core {

    // --- Enums and Structs for Sequence State & Parameters ---
    enum class SequenceStatus {
        WAITING,            // Received, awaiting scheduling
        PREFILLING,         // Currently being processed in a prefill batch
        DECODING,           // Currently being processed in a decode batch
        COMPLETED,          // Completed successfully
        ERROR               // An error occurred during processing
    };

    // --- Sequence Class ---
    class Sequence {
        public:
            const uint64_t sequence_id;
            SequenceStatus status;
            const uint64_t arrival_timestamp_ns;

            // --- Token & KV Cache State ---
            std::vector<int32_t> tokens;      // MUTABLE (prompt + generated)
            const size_t prompt_len;
            std::vector<uint32_t> page_table;

            const SamplingParams sampling_params; // Immutable for this sequence
            const LogitsParams logits_params;     // Immutable for this sequence
            const StopCriteria stop_criteria;     // Immutable for this sequence

            // --- Communication Handles ---
            const IPCHandles ipc_handles;

            std::atomic<bool> cancelled{false};

            // --- Helper Methods (const where possible) ---
            [[nodiscard]] size_t get_generation_len() const;
            [[nodiscard]] size_t get_logical_len() const;
            [[nodiscard]] bool is_finished() const;
            void append_token(int32_t token_id); // Non-const, modifies tokens
            void append_page(uint32_t page_id); // Non-const, modifies page_table
            [[nodiscard]] std::optional<uint32_t> get_physical_page(size_t logical_block_index) const;

            // --- Move & Copy Operations ---
            Sequence(const Sequence&) = delete;
            Sequence& operator=(const Sequence&) = delete;
            Sequence(Sequence&&) = default;
            Sequence& operator=(Sequence&&) = default;

        private:
    };

} // namespace pie_core
