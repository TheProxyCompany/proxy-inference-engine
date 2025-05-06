#include "ipc/shared_memory_manager.hpp"
#include <boost/interprocess/shared_memory_object.hpp>
#include <iostream>

namespace pie_core::ipc {

    SharedMemoryManager::SharedMemoryManager(
        const std::string& shm_name,
        size_t shm_size,
        bool create_if_not_exists)
        : shm_name_(shm_name),
          created_by_this_instance_(false)
    {
        try {
            if (create_if_not_exists) {
                segment_ = bip::managed_shared_memory(bip::create_only, shm_name_.c_str(), shm_size);
                created_by_this_instance_ = true;
                std::cout << "SharedMemoryManager: Created SHM segment '" << shm_name_ << "' of size " << shm_size << std::endl;
            } else {
                segment_ = bip::managed_shared_memory(bip::open_only, shm_name_.c_str());
                std::cout << "SharedMemoryManager: Opened existing SHM segment '" << shm_name_ << "'" << std::endl;
            }
        } catch (const bip::interprocess_exception& ex) {
            if (create_if_not_exists && ex.get_error_code() == bip::already_exists_error) {
                try {
                    segment_ = bip::managed_shared_memory(bip::open_only, shm_name_.c_str());
                    created_by_this_instance_ = false;
                    std::cout << "SharedMemoryManager: SHM segment '" << shm_name_ << "' already existed, opened it." << std::endl;
                } catch (const bip::interprocess_exception& ex_open) {
                    throw SharedMemoryError(
                        "Failed to open SHM segment '" + shm_name_ + "' after create attempt failed: " + ex_open.what());
                }
            } else {
                throw SharedMemoryError(
                    "Failed to create/open SHM segment '" + shm_name_ + "': " + ex.what());
            }
        }
    }

    SharedMemoryManager::~SharedMemoryManager() {
        if (created_by_this_instance_) {
            std::cout << "SharedMemoryManager: Removing SHM segment '" << shm_name_ << "'" << std::endl;
            if (!bip::shared_memory_object::remove(shm_name_.c_str())) {
                std::cerr << "Error: SharedMemoryManager: Failed to remove SHM segment '" << shm_name_
                          << "'. It might have been removed by another process or an error occurred." << std::endl;
            }
        }
    }

    void* SharedMemoryManager::allocate(size_t n_bytes) {
        if (n_bytes == 0) {
            return nullptr;
        }
        try {
            return segment_.allocate(n_bytes);
        } catch (const bip::bad_alloc& ex) {
            throw SharedMemoryError(
                "SHM allocation failed for " + std::to_string(n_bytes) + " bytes in segment '" +
                shm_name_ + "': " + ex.what() + ". Segment free memory: " + std::to_string(segment_.get_free_memory()));
        } catch (const std::exception& ex) {
            throw SharedMemoryError(
                "Unexpected error during SHM allocation in segment '" + shm_name_ + "': " + ex.what());
        }
    }

    void SharedMemoryManager::deallocate(void* ptr) {
        if (ptr == nullptr) {
            return;
        }
        try {
            segment_.deallocate(ptr);
        } catch (const std::exception& ex) {
            throw SharedMemoryError(
                "Error during SHM deallocation in segment '" + shm_name_ + "': " + ex.what() +
                ". Pointer was: " + std::to_string(reinterpret_cast<uintptr_t>(ptr)));
        }
    }

    void* SharedMemoryManager::get_segment_base_address() const {
        return segment_.get_address();
    }

} // namespace pie_core::ipc
