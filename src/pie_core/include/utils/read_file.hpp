#pragma once

#include <filesystem>
#include <fstream>
#include <system_error>
#include <vector>

/// \brief Read every byte of a file into memory.
///
/// * 100 % binary-safe (no newline translation)
/// * Uses `std::filesystem::file_size` for an O(1) size query
/// * Robust error handling via exceptions (no process‐killing `exit()`)
/// * Returns a `std::vector<std::byte>`—semantically “raw bytes”, not “text”
///
/// \throw std::system_error   if the file cannot be opened or sized
/// \throw std::runtime_error  if the read is incomplete
[[nodiscard]] inline std::vector<std::byte>
load_file_bytes(const std::filesystem::path& path)
{
    namespace fs = std::filesystem;

    // ── 1. Get size up front (fast, reuses OS metadata) ────────────────
    const auto size = fs::file_size(path);            // O(1) on POSIX/NTFS
    if (size == static_cast<std::uintmax_t>(-1))
        throw std::system_error(errno, std::generic_category(),
                                "file_size failed for " + path.string());

    // ── 2. Allocate the exact buffer once ─────────────────────────────
    std::vector<std::byte> buffer(size);

    // ── 3. Open + read with exception semantics ───────────────────────
    std::ifstream file(path, std::ios::binary);
    file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    if (size) {
        file.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(size));
    }

    return buffer;
}
