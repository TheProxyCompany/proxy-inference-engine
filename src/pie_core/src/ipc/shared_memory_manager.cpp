#include "ipc/shared_memory_manager.hpp"
#include <boost/interprocess/shared_memory_object.hpp>
#include <spdlog/spdlog.h>

namespace pie_core::ipc {

    SharedMemoryManager::SharedMemoryManager(
        const std::string& shm_name,
        size_t shm_size,
        bool create_if_not_exists)
        : shm_name_(shm_name),
          created_by_this_instance_(false)
    {
        spdlog::info("SharedMemoryManager: Initializing SHM segment '{}' with size {}", shm_name_, shm_size);
        try {
            if (create_if_not_exists) {
                segment_ = bip::managed_shared_memory(bip::create_only, shm_name_.c_str(), shm_size);
                created_by_this_instance_ = true;
                spdlog::info("SharedMemoryManager: Created SHM segment '{}' of size {}", shm_name_, shm_size);
            } else {
                segment_ = bip::managed_shared_memory(bip::open_only, shm_name_.c_str());
                spdlog::info("SharedMemoryManager: Opened existing SHM segment '{}'", shm_name_);
            }
            spdlog::debug("SharedMemoryManager: SHM segment '{}' base address: {:p}, free memory: {}",
                          shm_name_, segment_.get_address(), segment_.get_free_memory());
        } catch (const bip::interprocess_exception& ex) {
            if (create_if_not_exists && ex.get_error_code() == bip::already_exists_error) {
                spdlog::warn("SharedMemoryManager: SHM segment '{}' already exists, attempting to open it", shm_name_);
                try {
                    segment_ = bip::managed_shared_memory(bip::open_only, shm_name_.c_str());
                    created_by_this_instance_ = false;
                    spdlog::info("SharedMemoryManager: Successfully opened existing SHM segment '{}'", shm_name_);
                    spdlog::debug("SharedMemoryManager: SHM segment '{}' base address: {:p}, free memory: {}",
                                 shm_name_, segment_.get_address(), segment_.get_free_memory());
                } catch (const bip::interprocess_exception& ex_open) {
                    spdlog::error("SharedMemoryManager: Failed to open existing SHM segment '{}': {}",
                                 shm_name_, ex_open.what());
                    throw SharedMemoryError(
                        "Failed to open SHM segment '" + shm_name_ + "' after create attempt failed: " + ex_open.what());
                }
            } else {
                spdlog::error("SharedMemoryManager: Failed to create/open SHM segment '{}': {}",
                             shm_name_, ex.what());
                throw SharedMemoryError(
                    "Failed to create/open SHM segment '" + shm_name_ + "': " + ex.what());
            }
        }
    }

    SharedMemoryManager::~SharedMemoryManager() {
        if (created_by_this_instance_) {
            spdlog::info("SharedMemoryManager: Removing SHM segment '{}'", shm_name_);
            if (!bip::shared_memory_object::remove(shm_name_.c_str())) {
                spdlog::error("SharedMemoryManager: Failed to remove SHM segment '{}'. It might have been removed by another process or an error occurred.", shm_name_);
            } else {
                spdlog::debug("SharedMemoryManager: Successfully removed SHM segment '{}'", shm_name_);
            }
        } else {
            spdlog::debug("SharedMemoryManager: Not removing SHM segment '{}' as it was not created by this instance", shm_name_);
        }
    }

    void* SharedMemoryManager::allocate(size_t n_bytes) {
        if (n_bytes == 0) {
            spdlog::trace("SharedMemoryManager: Requested allocation of 0 bytes, returning nullptr");
            return nullptr;
        }

        try {
            spdlog::trace("SharedMemoryManager: Allocating {} bytes from segment '{}' (free memory: {})",
                         n_bytes, shm_name_, segment_.get_free_memory());
            void* ptr = segment_.allocate(n_bytes);
            spdlog::trace("SharedMemoryManager: Successfully allocated {} bytes at {:p} from segment '{}' (remaining free memory: {})",
                         n_bytes, ptr, shm_name_, segment_.get_free_memory());
            return ptr;
        } catch (const bip::bad_alloc& ex) {
            spdlog::error("SharedMemoryManager: SHM allocation failed for {} bytes in segment '{}': {}. Free memory: {}",
                         n_bytes, shm_name_, ex.what(), segment_.get_free_memory());
            throw SharedMemoryError(
                "SHM allocation failed for " + std::to_string(n_bytes) + " bytes in segment '" +
                shm_name_ + "': " + ex.what() + ". Segment free memory: " + std::to_string(segment_.get_free_memory()));
        } catch (const std::exception& ex) {
            spdlog::error("SharedMemoryManager: Unexpected error during SHM allocation of {} bytes in segment '{}': {}",
                         n_bytes, shm_name_, ex.what());
            throw SharedMemoryError(
                "Unexpected error during SHM allocation in segment '" + shm_name_ + "': " + ex.what());
        }
    }

    void SharedMemoryManager::deallocate(void* ptr) {
        if (ptr == nullptr) {
            spdlog::trace("SharedMemoryManager: Attempted to deallocate nullptr, ignoring");
            return;
        }

        try {
            size_t free_memory_before = segment_.get_free_memory();
            spdlog::trace("SharedMemoryManager: Deallocating memory at {:p} from segment '{}'", ptr, shm_name_);
            segment_.deallocate(ptr);
            size_t free_memory_after = segment_.get_free_memory();
            spdlog::trace("SharedMemoryManager: Successfully deallocated memory at {:p} from segment '{}' (freed {} bytes)",
                         ptr, shm_name_, free_memory_after - free_memory_before);
        } catch (const std::exception& ex) {
            spdlog::error("SharedMemoryManager: Error during SHM deallocation in segment '{}' at {:p}: {}",
                         shm_name_, ptr, ex.what());
            throw SharedMemoryError(
                "Error during SHM deallocation in segment '" + shm_name_ + "': " + ex.what() +
                ". Pointer was: " + std::to_string(reinterpret_cast<uintptr_t>(ptr)));
        }
    }

    void* SharedMemoryManager::get_segment_base_address() const {
        void* addr = segment_.get_address();
        spdlog::trace("SharedMemoryManager: Returning base address {:p} for segment '{}'", addr, shm_name_);
        return addr;
    }

} // namespace pie_core::ipc
