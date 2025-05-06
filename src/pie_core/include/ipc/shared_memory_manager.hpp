#pragma once

#include <string>
#include <memory>
#include <stdexcept>
#include <cstddef>

#include <boost/interprocess/managed_shared_memory.hpp>

namespace pie_core::ipc {

    namespace bip = boost::interprocess;

    class SharedMemoryError : public std::runtime_error {
    public:
        using std::runtime_error::runtime_error;
    };

    class SharedMemoryManager {
    public:
        /**
         * @brief Constructs or opens a shared memory segment.
         * @param shm_name The unique name of the shared memory segment (e.g., "/pie_bulk_data").
         * @param shm_size The total size of the segment if creating it. Ignored if opening an existing one.
         * @param create_if_not_exists If true, creates the segment if it doesn't exist.
         *                             If false, only attempts to open an existing segment.
         */
        SharedMemoryManager(const std::string& shm_name, size_t shm_size, bool create_if_not_exists);

        /**
         * @brief Destructor. Cleans up resources.
         *        If this instance created the segment, it will be removed (shm_unlink).
         */
        ~SharedMemoryManager();

        /**
         * @brief Allocates a block of memory within the shared segment.
         * @param n_bytes The number of bytes to allocate.
         * @return A raw pointer to the allocated block within the shared memory segment.
         * @throws SharedMemoryError if allocation fails (e.g., out of memory).
         *         The pointer is relative to the process's mapped view of the segment.
         */
        void* allocate(size_t n_bytes);

        /**
         * @brief Deallocates a previously allocated block of memory.
         * @param ptr A pointer previously returned by allocate().
         *            The pointer must be valid and point within this managed segment.
         */
        void deallocate(void* ptr);

        /**
         * @brief Gets the base address of the mapped shared memory segment in this process's address space.
         * @return Raw pointer to the base of the mapped segment.
         */
        void* get_segment_base_address() const;

        // --- Rule of 5/6: Prevent copying and assignment ---
        SharedMemoryManager(const SharedMemoryManager&) = delete;
        SharedMemoryManager& operator=(const SharedMemoryManager&) = delete;
        SharedMemoryManager(SharedMemoryManager&&) = delete;
        SharedMemoryManager& operator=(SharedMemoryManager&&) = delete;

    private:
        std::string shm_name_;
        bool created_by_this_instance_;

        bip::managed_shared_memory segment_;
    };

} // namespace pie_core::ipc
