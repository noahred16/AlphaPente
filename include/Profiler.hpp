#ifndef PROFILER_HPP
#define PROFILER_HPP

// ============================================================================
// Profiling Configuration
// ============================================================================
// Define ENABLE_PROFILING to enable profiling. Defaults to disabled.
// Enable via: cmake -DENABLE_PROFILING=ON or #define ENABLE_PROFILING before including

#define ENABLE_PROFILING  // Uncomment to enable, or pass -DENABLE_PROFILING. Comment out to disable.

#ifdef ENABLE_PROFILING

#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <mutex>

// ============================================================================
// Profiler - Accumulates timing data across many function calls
// ============================================================================

class Profiler {
public:
    struct SectionStats {
        uint64_t callCount = 0;
        double totalTimeNs = 0.0;  // Nanoseconds for precision
    };

    // Get singleton instance
    static Profiler& instance() {
        static Profiler profiler;
        return profiler;
    }

    // Record a timing measurement for a section
    void record(const std::string& section, double durationNs) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto& stats = sections_[section];
        stats.callCount++;
        stats.totalTimeNs += durationNs;
    }

    // Reset all accumulated data
    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        sections_.clear();
    }

    // Print formatted report
    void printReport() const {
        std::lock_guard<std::mutex> lock(mutex_);

        if (sections_.empty()) {
            std::cout << "\n=== Profiler Report ===\n";
            std::cout << "No profiling data collected.\n";
            return;
        }

        // Collect and sort by total time (descending)
        std::vector<std::pair<std::string, SectionStats>> sorted(
            sections_.begin(), sections_.end());
        std::sort(sorted.begin(), sorted.end(),
            [](const auto& a, const auto& b) {
                return a.second.totalTimeNs > b.second.totalTimeNs;
            });

        // Calculate total time for percentage
        double grandTotal = 0.0;
        for (const auto& [name, stats] : sorted) {
            grandTotal += stats.totalTimeNs;
        }

        // Print header
        std::cout << "\n";
        std::cout << "================================================================================\n";
        std::cout << "                              PROFILER REPORT                                   \n";
        std::cout << "================================================================================\n";
        std::cout << std::left << std::setw(28) << "Section"
                  << std::right << std::setw(14) << "Total Time"
                  << std::setw(10) << "   %"
                  << std::setw(14) << "Calls"
                  << std::setw(14) << "Avg/Call"
                  << "\n";
        std::cout << std::string(80, '-') << "\n";

        // Print each section
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
        std::cout << "================================================================================\n\n";
    }

private:
    Profiler() = default;
    ~Profiler() = default;

    // Non-copyable
    Profiler(const Profiler&) = delete;
    Profiler& operator=(const Profiler&) = delete;

    mutable std::mutex mutex_;
    std::unordered_map<std::string, SectionStats> sections_;
};

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

    // Non-copyable, non-movable
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
