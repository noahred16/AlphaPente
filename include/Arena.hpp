#ifndef ARENA_HPP
#define ARENA_HPP

#include <cstdlib>
#include <cstdio>
#include <new>
#include <utility>

// ============================================================================
// Arena Allocator - O(1) bulk deallocation via offset reset
// ============================================================================

class Arena {
  public:
    explicit Arena(size_t size) : size_(size), offset_(0), memory_(nullptr) {
        size_t avail = effectiveMemLimitBytes();
        if (avail > 0 && size_ > avail) {
            std::fprintf(stderr,
                "WARNING: Arena (%.1f GB) exceeds available RAM (%.1f GB). "
                "May OOM-kill under load. Re-run scripts/detect_system_specs.sh "
                "or lower ARENA_SIZE_GB.\n",
                size_ / (1024.0 * 1024.0 * 1024.0),
                avail / (1024.0 * 1024.0 * 1024.0));
        }
        memory_ = static_cast<char *>(std::aligned_alloc(64, size_));
        if (!memory_) {
            std::fprintf(stderr, "FATAL: Failed to allocate arena of %.1f GB. "
                "Not enough system memory (check `free -h`).\n",
                size_ / (1024.0 * 1024.0 * 1024.0));
            throw std::bad_alloc();
        }
    }

    ~Arena() { std::free(memory_); }

    Arena(const Arena &) = delete;
    Arena &operator=(const Arena &) = delete;

    template <typename T> T *allocate(size_t count = 1) {
        size_t alignment = alignof(T);
        size_t alignedOffset = (offset_ + alignment - 1) & ~(alignment - 1);
        size_t totalBytes = sizeof(T) * count;

        if (alignedOffset + totalBytes > size_)
            return nullptr;

        T *ptr = reinterpret_cast<T *>(memory_ + alignedOffset);
        offset_ = alignedOffset + totalBytes;
        return ptr;
    }

    void *allocateBytes(size_t bytes, size_t alignment) {
        size_t alignedOffset = (offset_ + alignment - 1) & ~(alignment - 1);
        if (alignedOffset + bytes > size_) return nullptr;
        void *ptr = memory_ + alignedOffset;
        offset_ = alignedOffset + bytes;
        return ptr;
    }

    void reset() { offset_ = 0; }

    void swap(Arena &other) {
        std::swap(size_, other.size_);
        std::swap(offset_, other.offset_);
        std::swap(memory_, other.memory_);
    }

    size_t bytesUsed() const { return offset_; }
    size_t bytesRemaining() const { return size_ - offset_; }
    size_t totalSize() const { return size_; }
    double utilizationPercent() const { return 100.0 * offset_ / size_; }

  private:
    size_t size_;
    size_t offset_;
    char *memory_;

    // Returns cgroup v2 limit (containers) or MemAvailable (bare metal), 0 on failure.
    static size_t effectiveMemLimitBytes() {
        FILE *f = std::fopen("/sys/fs/cgroup/memory.max", "r");
        if (f) {
            unsigned long long limit = 0;
            int matched = std::fscanf(f, "%llu", &limit);
            std::fclose(f);
            if (matched == 1 && limit > 0) return static_cast<size_t>(limit);
        }
        f = std::fopen("/proc/meminfo", "r");
        if (!f) return 0;
        char line[128];
        size_t avail = 0;
        while (std::fgets(line, sizeof(line), f)) {
            if (std::sscanf(line, "MemAvailable: %zu kB", &avail) == 1) break;
        }
        std::fclose(f);
        return avail * 1024;
    }
};

#endif // ARENA_HPP
