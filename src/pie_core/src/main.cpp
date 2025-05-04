// src/pie_core/src/engine_main.cpp
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <sys/mman.h> // mmap, shm_open
#include <sys/stat.h> // For mode constants
#include <fcntl.h>    // For O_* constants
#include <unistd.h>   // ftruncate, close
#include <sys/event.h> // kqueue
#include <csignal>    // signal handling

#include "ipc/ipc_defs.hpp"

// --- Global variables (simplify for now, use classes later) ---
void* shm_ptr = nullptr;
int kqueue_fd = -1;
int shm_fd = -1;
volatile bool running = true;
// ---

void signal_handler(int signum) {
    std::cout << "Signal (" << signum << ") received. Shutting down." << std::endl;
    running = false;
}

// --- Placeholder for Scheduler Thread ---
void scheduler_loop() {
    std::cout << "Scheduler thread started." << std::endl;
    pie_core::ipc::RequestSlot* slots = static_cast<pie_core::ipc::RequestSlot*>(shm_ptr);

    while(running) {
        // --- TODO: Milestone 6: Replace this polling with kqueue wait ---
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Poll interval

        // --- Milestone 3: Implement polling consumer logic ---
        std::cout << "Scheduler checking for requests..." << std::endl;
         // Simplified check - a real implementation needs careful indexing
        for (size_t i = 0; i < pie_core::ipc::NUM_SLOTS; ++i) {
             // Naive scan - better: use consumer_idx & check producer_idx
             uint32_t expected_state = 2; // READY
             if (slots[i].state.compare_exchange_strong(expected_state, 0, std::memory_order_acq_rel)) {
                  // Successfully claimed a READY slot and marked it FREE (0)
                  std::cout << "Scheduler received request ID: "
                            << slots[i].request_id << std::endl;
                  // TODO: Process the request data (copy out, add to internal queue)
             }
        }
        // --- End Milestone 3 ---
    }
    std::cout << "Scheduler thread exiting." << std::endl;
}
// ---

int main(int argc, char *argv[]) {
    std::cout << "Starting PIE Engine Process..." << std::endl;
    std::cout << "argc: " << argc << std::endl;
    if (argc > 1) {
        std::cout << "argv[0]: " << argv[0] << std::endl;
    }

    // --- Milestone 1 & 2: Setup Shared Memory & kqueue ---
    // Create/Open Shared Memory
    shm_fd = shm_open(pie_core::ipc::SHM_NAME, O_CREAT | O_RDWR, 0666);
    if (shm_fd == -1) {
        perror("shm_open");
        return 1;
    }
    if (ftruncate(shm_fd, pie_core::ipc::SHM_SIZE) == -1) {
        perror("ftruncate");
        close(shm_fd);
        shm_unlink(pie_core::ipc::SHM_NAME); // Clean up on error
        return 1;
    }
    shm_ptr = mmap(nullptr, pie_core::ipc::SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (shm_ptr == MAP_FAILED) {
        perror("mmap");
        close(shm_fd);
        shm_unlink(pie_core::ipc::SHM_NAME);
        return 1;
    }
    std::cout << "Shared memory '" << pie_core::ipc::SHM_NAME << "' created/mapped." << std::endl;

    // Create kqueue
    kqueue_fd = kqueue();
    if (kqueue_fd == -1) {
        perror("kqueue");
        // Cleanup shm
        munmap(shm_ptr, pie_core::ipc::SHM_SIZE);
        close(shm_fd);
        shm_unlink(pie_core::ipc::SHM_NAME);
        return 1;
    }

    // Register user event filter
    struct kevent change;
    EV_SET(&change, 1, EVFILT_USER, EV_ADD | EV_CLEAR, 0, 0, nullptr);
    if (kevent(kqueue_fd, &change, 1, nullptr, 0, nullptr) == -1) {
        perror("kevent register");
        // Cleanup kqueue and shm
        close(kqueue_fd);
        munmap(shm_ptr, pie_core::ipc::SHM_SIZE);
        close(shm_fd);
        shm_unlink(pie_core::ipc::SHM_NAME);
        return 1;
    }
    std::cout << "kqueue created (fd=" << kqueue_fd << ") and user event registered." << std::endl;
    // --- End Milestone 1 & 2 ---

    // TODO: Initialize PageAllocator, Scheduler

    // Start Scheduler Thread
    std::thread scheduler_thread(scheduler_loop);

    // Handle signals for graceful shutdown
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // Keep main thread alive (or do other work)
    scheduler_thread.join(); // Wait for scheduler to finish

    // Cleanup
    std::cout << "Cleaning up resources..." << std::endl;
    close(kqueue_fd);
    munmap(shm_ptr, pie_core::ipc::SHM_SIZE);
    close(shm_fd);
    shm_unlink(pie_core::ipc::SHM_NAME); // Remove shared memory object name
    std::cout << "PIE Engine Process finished." << std::endl;

    return 0;
}
