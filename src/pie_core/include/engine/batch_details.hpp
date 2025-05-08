#pragma once

#include <cstdint>
#include <mlx/mlx.h>

namespace mx = mlx::core;

namespace pie_core::engine {

    /**
     * @brief Enum to specify which attention implementation to use
     */
    enum class AttentionType {
        STANDARD, // Use standard MLX scaled_dot_product_attention
        PAGED     // Use custom paged attention kernel
    };

    struct BatchDetails {

        BatchDetails()
            : token_ids(mx::array({})),
              positions(mx::array({})),
              sequence_ids(std::vector<uint64_t>()),
              input_lengths(std::vector<int32_t>()),
              context_lengths(std::vector<int32_t>()),
              consolidated_block_table(mx::array({})),
              num_prefill_sequences(0),
              num_decode_sequences(0),
              total_tokens_in_step(0),
              attention_type(AttentionType::STANDARD) // Default to standard for baseline testing
        {}

        /**
         * @brief Concatenated token IDs for all sequences in the batch for this step.
         * For prefill sequences, this includes the chunk of prompt tokens.
         * For decode sequences, this includes the single token to be processed.
         * Shape: [total_tokens_in_step]
         */
        mx::array token_ids;

        /**
         * @brief Corresponding position IDs for each token in `token_ids`.
         * Takes into account the sequence's context length for RoPE calculation.
         * Shape: [total_tokens_in_step]
         */
        mx::array positions;


        // --- Sequence Mapping & Length Information ---

        /**
         * @brief The unique IDs of the sequences included in this batch.
         * The order MUST correspond to how sequences' data is laid out in
         * `token_ids`, `positions`, `consolidated_block_table`, etc.
         */
        std::vector<uint64_t> sequence_ids;

        /**
         * @brief The number of *new* tokens being processed for each sequence in this step.
         * This is typically > 1 for prefill chunks and == 1 for decode steps.
         * The sum of this vector equals `total_tokens_in_step`.
         * Size: [num_sequences_in_batch] (where num_sequences = sequence_ids.size())
         */
        std::vector<int32_t> input_lengths;

        /**
         * @brief The logical length of each sequence *before* this step was processed.
         * This is crucial for KV cache indexing/masking and RoPE offsets.
         * Size: [num_sequences_in_batch]
         */
        std::vector<int32_t> context_lengths;


        // --- Paged Attention Data ---

        /**
         * @brief The consolidated block table mapping logical blocks to physical page IDs
         *        for ALL sequences in the batch.
         *
         * The exact format (e.g., 1D flattened, 2D [seq_idx, block_idx]) and
         * indexing scheme MUST be co-designed and synchronized with the specific
         * requirements of the Paged Attention Metal kernel implementation.
         * It might map a batch-local sequence index and a logical block index
         * to a physical page ID managed by the PageAllocator.
         */
        mx::array consolidated_block_table;

        // --- Batch Metadata ---

        /** @brief Number of sequences in the PREFILLING state within this batch. */
        size_t num_prefill_sequences = 0;

        /** @brief Number of sequences in the DECODING state within this batch. */
        size_t num_decode_sequences = 0;

        /**
         * @brief Total number of tokens being processed in this specific step.
         * Should equal `token_ids.shape[0]` and `sum(input_lengths)`.
         */
        size_t total_tokens_in_step = 0;

        /**
         * @brief The type of attention mechanism to use for this batch
         */
        AttentionType attention_type = AttentionType::STANDARD;
    };
} // namespace pie_core::engine
