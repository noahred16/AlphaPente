#pragma once

#include <chrono>

namespace utils {

class Timer {
public:
    Timer();
    ~Timer() = default;
    
    void start();
    void stop();
    double getElapsedMs() const;
    double getElapsedSeconds() const;

private:
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point end_time_;
    bool running_;
};

} // namespace utils