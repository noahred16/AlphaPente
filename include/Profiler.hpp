#ifndef PROFILER_HPP
#define PROFILER_HPP

// ============================================================================
// Profiling Configuration
// ============================================================================
// Define ENABLE_PROFILING to enable profiling. Defaults to disabled.
// Enable via: cmake -DENABLE_PROFILING=ON or #define ENABLE_PROFILING before including

// #define ENABLE_PROFILING  // Uncomment to enable, or pass -DENABLE_PROFILING. Comment out to disable.

#ifdef ENABLE_PROFILING

#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <mutex>
#include <memory>
#include <thread>
#include <sstream>

// ============================================================================
// Profiler - Lock-free per-thread accumulation, mutex only on registration
// ============================================================================

class Profiler {
public:
    struct SectionStats {
        uint64_t callCount = 0;
        double totalTimeNs = 0.0;
    };

    struct ThreadData {
        std::thread::id threadId;
        std::string label;
        std::unordered_map<std::string, SectionStats> sections;
    };

    static Profiler& instance() {
        static Profiler profiler;
        return profiler;
    }

    // Fast path: write to thread-local map, no mutex
    void record(const std::string& section, double durationNs) {
        ThreadData* td = myData_;
        if (!td) {
            td = registerThread();
        }
        auto& stats = td->sections[section];
        stats.callCount++;
        stats.totalTimeNs += durationNs;
    }

    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        allThreadData_.clear();
        myData_ = nullptr;  // Only clears the calling thread's pointer
    }

    void printReport() const {
        std::lock_guard<std::mutex> lock(mutex_);

        if (allThreadData_.empty()) {
            std::cout << "\n=== Profiler Report ===\n";
            std::cout << "No profiling data collected.\n";
            return;
        }

        // ---- Aggregate table ----
        std::unordered_map<std::string, SectionStats> aggregate;
        for (const auto& td : allThreadData_) {
            for (const auto& [name, stats] : td->sections) {
                auto& agg = aggregate[name];
                agg.callCount += stats.callCount;
                agg.totalTimeNs += stats.totalTimeNs;
            }
        }

        std::vector<std::pair<std::string, SectionStats>> sorted(
            aggregate.begin(), aggregate.end());
        std::sort(sorted.begin(), sorted.end(),
            [](const auto& a, const auto& b) {
                return a.second.totalTimeNs > b.second.totalTimeNs;
            });

        double grandTotal = 0.0;
        for (const auto& [name, stats] : sorted) {
            grandTotal += stats.totalTimeNs;
        }

        std::cout << "\n";
        std::cout << "================================================================================\n";
        std::cout << "                         PROFILER REPORT (AGGREGATE)                             \n";
        std::cout << "================================================================================\n";
        std::cout << std::left << std::setw(28) << "Section"
                  << std::right << std::setw(14) << "Total Time"
                  << std::setw(10) << "   %"
                  << std::setw(14) << "Calls"
                  << std::setw(14) << "Avg/Call"
                  << "\n";
        std::cout << std::string(80, '-') << "\n";

        for (const auto& [name, stats] : sorted) {
            double totalMs = stats.totalTimeNs / 1e6;
            double avgNs = stats.callCount > 0 ? stats.totalTimeNs / stats.callCount : 0.0;
            double pct = grandTotal > 0 ? (stats.totalTimeNs / grandTotal) * 100.0 : 0.0;

            std::cout << std::left << std::setw(28) << name
                      << std::right << std::setw(10) << std::fixed << std::setprecision(2) << totalMs << " ms"
                      << std::setw(8) << std::fixed << std::setprecision(1) << pct << " %"
                      << std::setw(14) << stats.callCount
                      << std::setw(10) << std::fixed << std::setprecision(1) << avgNs << " ns"
                      << "\n";
        }

        std::cout << std::string(80, '-') << "\n";
        std::cout << std::left << std::setw(28) << "TOTAL"
                  << std::right << std::setw(10) << std::fixed << std::setprecision(2) << (grandTotal / 1e6) << " ms"
                  << "\n";
        std::cout << "================================================================================\n";

        // ---- Per-thread breakdown ----
        if (allThreadData_.size() > 1) {
            std::cout << "\n";
            std::cout << "================================================================================\n";
            std::cout << "                          PER-THREAD BREAKDOWN                                   \n";
            std::cout << "================================================================================\n";

            for (size_t t = 0; t < allThreadData_.size(); ++t) {
                const auto& td = allThreadData_[t];

                double threadTotal = 0.0;
                for (const auto& [name, stats] : td->sections) {
                    threadTotal += stats.totalTimeNs;
                }

                std::cout << "\n--- " << td->label << " (total: "
                          << std::fixed << std::setprecision(2) << (threadTotal / 1e6)
                          << " ms) ---\n";

                // Sort this thread's sections by time
                std::vector<std::pair<std::string, SectionStats>> threadSorted(
                    td->sections.begin(), td->sections.end());
                std::sort(threadSorted.begin(), threadSorted.end(),
                    [](const auto& a, const auto& b) {
                        return a.second.totalTimeNs > b.second.totalTimeNs;
                    });

                std::cout << std::left << std::setw(28) << "Section"
                          << std::right << std::setw(14) << "Total Time"
                          << std::setw(10) << "   %"
                          << std::setw(14) << "Calls"
                          << "\n";
                std::cout << std::string(66, '-') << "\n";

                for (const auto& [name, stats] : threadSorted) {
                    double totalMs = stats.totalTimeNs / 1e6;
                    double pct = threadTotal > 0 ? (stats.totalTimeNs / threadTotal) * 100.0 : 0.0;

                    std::cout << std::left << std::setw(28) << name
                              << std::right << std::setw(10) << std::fixed << std::setprecision(2) << totalMs << " ms"
                              << std::setw(8) << std::fixed << std::setprecision(1) << pct << " %"
                              << std::setw(14) << stats.callCount
                              << "\n";
                }
            }

            std::cout << "================================================================================\n";
        }

        std::cout << "\n";
    }

private:
    Profiler() = default;
    ~Profiler() = default;
    Profiler(const Profiler&) = delete;
    Profiler& operator=(const Profiler&) = delete;

    // Called once per thread on first record() call
    ThreadData* registerThread() {
        std::lock_guard<std::mutex> lock(mutex_);
        auto td = std::make_unique<ThreadData>();
        td->threadId = std::this_thread::get_id();
        std::ostringstream oss;
        oss << "Thread " << allThreadData_.size();
        td->label = oss.str();
        ThreadData* raw = td.get();
        allThreadData_.push_back(std::move(td));
        myData_ = raw;
        return raw;
    }

    mutable std::mutex mutex_;
    std::vector<std::unique_ptr<ThreadData>> allThreadData_;
    static thread_local ThreadData* myData_;
};

// Definition of the thread_local pointer (inline for header-only)
inline thread_local Profiler::ThreadData* Profiler::myData_ = nullptr;

// ============================================================================
// ScopedTimer - RAII helper that records timing when it goes out of scope
// ============================================================================

class ScopedTimer {
public:
    explicit ScopedTimer(const std::string& section)
        : section_(section)
        , start_(std::chrono::high_resolution_clock::now()) {
    }

    ~ScopedTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto durationNs = std::chrono::duration<double, std::nano>(end - start_).count();
        Profiler::instance().record(section_, durationNs);
    }

    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
    std::string section_;
    std::chrono::high_resolution_clock::time_point start_;
};

// ============================================================================
// Macros - Active when profiling enabled
// ============================================================================

#define PROFILE_FUNCTION() ScopedTimer _profiler_timer_(__func__)
#define PROFILE_SCOPE(name) ScopedTimer _profiler_timer_##__LINE__(name)

#else // ENABLE_PROFILING not defined

// ============================================================================
// Stub Profiler - No-op when profiling disabled
// ============================================================================

class Profiler {
public:
    static Profiler& instance() {
        static Profiler profiler;
        return profiler;
    }
    void printReport() const {}  // No-op
    void reset() {}              // No-op
};

// ============================================================================
// Macros - No-op when profiling disabled
// ============================================================================

#define PROFILE_FUNCTION() ((void)0)
#define PROFILE_SCOPE(name) ((void)0)

#endif // ENABLE_PROFILING

#endif // PROFILER_HPP
