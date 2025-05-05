#pragma once

#include <cstddef>

// Guard Tracy includes and define Tracy-enabled macros only when TRACY_ENABLE is defined
#if defined(TRACY_ENABLE)

// Tracy includes - also guarded to prevent inclusion issues if library isn't available
#include <tracy/Tracy.hpp>

// Define macros for Tracy zones
#define PIE_PROFILE_ZONE(name) ZoneScopedN(name)
#define PIE_PROFILE_FUNCTION() ZoneScoped
#define PIE_PROFILE_FRAME(name) FrameMarkNamed(name)
#define PIE_PROFILE_FRAME_BEGIN(name) FrameMarkStart(name)
#define PIE_PROFILE_FRAME_END(name) FrameMarkEnd(name)
#define PIE_PROFILE_THREAD(name) tracy::SetThreadName(name)
#define PIE_PROFILE_PLOT(name, value) TracyPlot(name, value)
#define PIE_PROFILE_MEMORY(ptr, size) TracyAlloc(ptr, size)
#define PIE_PROFILE_MEMORY_FREE(ptr) TracyFree(ptr)
#define PIE_PROFILE_MUTEX(mtx) TracyLockableN(mtx, #mtx)
#define PIE_PROFILE_LOCK(mtx) LockMark(mtx)
#define PIE_PROFILE_MESSAGE(msg) TracyMessage(msg, strlen(msg))
#define PIE_PROFILE_FIBER(name) tracy::SetThreadName(name)

// Helper for profiling GPU work - only when GPU profiling is also enabled
#if defined(PIE_TRACY_PROFILE_GPU)
#define PIE_PROFILE_GPU_ZONE(name, stream) TracyGpuZone(name, stream)
#define PIE_PROFILE_GPU_COLLECT(stream) TracyGpuCollect(stream)
#else
#define PIE_PROFILE_GPU_ZONE(name, stream)
#define PIE_PROFILE_GPU_COLLECT(stream)
#endif

#else // TRACY_ENABLE not defined

// No-op implementations when Tracy is disabled
#define PIE_PROFILE_ZONE(name)
#define PIE_PROFILE_FUNCTION()
#define PIE_PROFILE_FRAME(name)
#define PIE_PROFILE_FRAME_BEGIN(name)
#define PIE_PROFILE_FRAME_END(name)
#define PIE_PROFILE_THREAD(name)
#define PIE_PROFILE_PLOT(name, value)
#define PIE_PROFILE_MEMORY(ptr, size)
#define PIE_PROFILE_MEMORY_FREE(ptr)
#define PIE_PROFILE_MUTEX(mtx)
#define PIE_PROFILE_LOCK(mtx)
#define PIE_PROFILE_MESSAGE(msg)
#define PIE_PROFILE_FIBER(name)
#define PIE_PROFILE_GPU_ZONE(name, stream)
#define PIE_PROFILE_GPU_COLLECT(stream)

#endif // TRACY_ENABLE

namespace pie_core {
namespace profiling {

// Class to wrap a mutex for Tracy profiling
template<typename Mutex>
class ProfiledMutex {
public:
    ProfiledMutex() = default;
    explicit ProfiledMutex(const char*) {} // Name parameter not used in simplified implementation

    void lock() {
#if defined(TRACY_ENABLE)
        PIE_PROFILE_ZONE("Mutex::lock");
#endif
        m_mutex.lock();
    }

    void unlock() {
#if defined(TRACY_ENABLE)
        PIE_PROFILE_ZONE("Mutex::unlock");
#endif
        m_mutex.unlock();
    }

    bool try_lock() {
#if defined(TRACY_ENABLE)
        PIE_PROFILE_ZONE("Mutex::try_lock");
#endif
        return m_mutex.try_lock();
    }

private:
    Mutex m_mutex;
};

// Utility class to mark a frame in a scope
class ScopedFrameMark {
public:
    explicit ScopedFrameMark(const char* name) : m_name(name) {
        PIE_PROFILE_FRAME_BEGIN(m_name);
    }

    ~ScopedFrameMark() {
        PIE_PROFILE_FRAME_END(m_name);
    }

private:
    [[maybe_unused]] const char* m_name;
};

// Simplified memory tracking helpers compatible with smart pointers
template<typename T>
inline T* profile_new(size_t count = 1, const char* = nullptr) {
    PIE_PROFILE_ZONE("profile_new[]");
    T* ptr = new T[count];
#if defined(TRACY_ENABLE)
    TracyAlloc(ptr, sizeof(T) * count);
#endif
    return ptr;
}

template<typename T>
inline void profile_delete(T* ptr) {
    PIE_PROFILE_ZONE("profile_delete[]");
#if defined(TRACY_ENABLE)
    TracyFree(ptr);
#endif
    delete[] ptr;
}

template<typename T, typename... Args>
inline T* profile_new_object(Args&&... args) {
    PIE_PROFILE_ZONE("profile_new_object");
    T* ptr = new T(std::forward<Args>(args)...);
#if defined(TRACY_ENABLE)
    TracyAlloc(ptr, sizeof(T));
#endif
    return ptr;
}

template<typename T>
inline void profile_delete_object(T* ptr) {
    PIE_PROFILE_ZONE("profile_delete_object");
#if defined(TRACY_ENABLE)
    TracyFree(ptr);
#endif
    delete ptr;
}

}} // namespace pie_core::profiling
