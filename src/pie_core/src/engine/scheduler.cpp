#include <deque>
#include <vector>
#include <list>
#include <unordered_map>
#include <mutex>
#include <random>
#include <chrono>
#include <spdlog/spdlog.h>

#include "engine/scheduler.hpp"
#include "engine/page_allocator.hpp"
#include "models/imodel.hpp"
#include "sequence/sequence.hpp"
#include "engine/batch_details.hpp"
#include "samplers/sampler_factory.hpp"
#include "logit_processors/logit_processor_factory.hpp"
#include <boost/lockfree/spsc_queue.hpp>

namespace mx = mlx::core;

namespace pie_core::engine {

    using WaitingSequenceQueue = boost::lockfree::spsc_queue<
        std::unique_ptr<sequence::Sequence>,
        boost::lockfree::capacity<1024>
    >;

    struct Scheduler::SchedulerImpl {
        PageAllocator& allocator_;
        std::unique_ptr<IModel> model_;
        const size_t max_num_seqs_;
        const size_t max_tokens_in_batch_;
        WaitingSequenceQueue waiting_sequences_queue_;
    };

}
