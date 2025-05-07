#include "ipc/request_reader.hpp"
#include "engine/raw_request.hpp"

#include <sys/event.h>      // kqueue
#include <unistd.h>
#include <spdlog/spdlog.h>

namespace pie_core::ipc {

    RequestReader::RequestReader(
        RawRequestQueue& output_queue,
        SharedMemoryManager& shm_manager,
        const std::string& request_shm_name,
        int kernel_event_fd
    ):
        output_queue_(output_queue),
        shm_manager_(shm_manager),
        kernel_event_fd_(kernel_event_fd)
    {
        if (!initialize_ipc_resources(request_shm_name)) {
            throw std::runtime_error("Failed to initialize IPC resources");
        }
    }

    RequestReader::~RequestReader() {
        stop();
        cleanup_ipc_resources();
    }

    void RequestReader::run() {
        running_.store(true, std::memory_order_release);
        while (running_.load(std::memory_order_acquire)) {
            if (!wait_for_notification()) {          // timeout or error
                continue;
            }
            process_incoming_requests();
        }
    }

    void RequestReader::stop() {
        running_.store(false, std::memory_order_release);
    }

    bool RequestReader::initialize_ipc_resources(const std::string& name) {
        request_shm_fd_ = shm_open(
            name.c_str(),
            O_RDWR,
            0
        );
        if (request_shm_fd_ < 0) {
            perror("shm_open");
            return false;
        }

        request_shm_map_ptr_ = mmap(
            nullptr,
            REQUEST_QUEUE_SHM_SIZE,
            PROT_READ|PROT_WRITE,
            MAP_SHARED,
            request_shm_fd_,
            0
        );
        if (request_shm_map_ptr_ == MAP_FAILED) {
            perror("mmap");
            return false;
        }

        request_slots_ = static_cast<RequestSlot*>(request_shm_map_ptr_);
        request_queue_control_ = reinterpret_cast<RequestQueueControl*>(
            static_cast<char*>(request_shm_map_ptr_)
            + REQUEST_QUEUE_NUM_SLOTS*sizeof(RequestSlot)
        );

        bulk_data_map_ptr_ = shm_manager_.get_segment_base_address();

        return true;
    }

    void RequestReader::cleanup_ipc_resources() {
        if (request_shm_map_ptr_) {
            munmap(request_shm_map_ptr_, REQUEST_QUEUE_SHM_SIZE);
        }
        if (request_shm_fd_ >= 0) {
            close(request_shm_fd_);
        }
    }

    bool RequestReader::wait_for_notification() {
        /* Plain kqueue one-shot wait */
        struct kevent Kev;
        EV_SET(&Kev, /*ident*/kernel_event_fd_, EVFILT_READ,
               EV_ADD | EV_ENABLE | EV_CLEAR, 0, 0, nullptr);

        int kq = kqueue();
        if (kq == -1) {
            perror("kqueue");
            return false;
        }

        struct kevent out;
        int nevents = kevent(kq, &Kev, 1, &out, 1, nullptr);
        close(kq);
        return nevents > 0;
    }

    void RequestReader::process_incoming_requests() {
        auto prod = request_queue_control_->producer_idx.load(std::memory_order_acquire);
        auto cons = request_queue_control_->consumer_idx.load(std::memory_order_relaxed);

        while (cons != prod) {
            RequestSlot& slot = request_slots_[cons % REQUEST_QUEUE_NUM_SLOTS];

            RequestState expected = RequestState::READY;
            if (!slot.state.compare_exchange_strong(expected,
                                                    RequestState::READING,
                                                    std::memory_order_acq_rel)) {
                break;
            }

            /* ---- copy / build RawRequestData ---- */
            auto raw = std::make_unique<engine::RawRequestData>();
            raw->request_id           = slot.request_id;
            raw->arrival_timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            raw->_shm_prompt_offset   = slot.prompt_shm_offset;
            raw->_shm_prompt_size     = slot.prompt_shm_size;
            raw->prompt_payload       = read_prompt_string(slot.prompt_shm_offset, slot.prompt_shm_size);
            raw->sampling_params      = slot.sampling_params;
            raw->logits_params        = slot.logits_params;
            raw->stop_criteria        = slot.stop_criteria;
            raw->ipc_handles          = slot.ipc_handles;
            raw->tool_schemas_str     = slot.tool_schemas_str;
            raw->response_format_str  = slot.response_format_str;

            if (!output_queue_.push(std::move(raw))) {
                spdlog::error("RequestReader: RawRequestQueue full – dropping request {}", slot.request_id);
                slot.state.store(RequestState::FREE, std::memory_order_release);
                break;      // prevent producer from racing with us
            }

            /* mark slot FREE */
            slot.state.store(RequestState::FREE, std::memory_order_release);
            ++cons;
            request_queue_control_->consumer_idx.store(cons, std::memory_order_release);
        }
    }

    std::string RequestReader::read_prompt_string(uint64_t offset, uint64_t size) {
        char* base = static_cast<char*>(bulk_data_map_ptr_);
        return std::string(base + offset, size);
    }


}
