#include "sequence/sequence.hpp"

namespace pie_core::sequence {

    Sequence::Sequence(
        uint64_t sequence_id,
        SequenceStatus status,
        uint64_t arrival_timestamp_ns,
        const std::vector<int32_t>& tokens,
        size_t prompt_len,
        const SamplingParams& sampling_params,
        const LogitsParams& logits_params,
        const StopCriteria& stop_criteria,
        const IPCHandles& ipc_handles
    ):
        sequence_id(sequence_id),
        status(status),
        arrival_timestamp_ns(arrival_timestamp_ns),
        tokens(tokens),
        prompt_len(prompt_len),
        sampling_params(sampling_params),
        logits_params(logits_params),
        stop_criteria(stop_criteria),
        ipc_handles(ipc_handles)
    {

    }
}
