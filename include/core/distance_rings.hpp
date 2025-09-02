#pragma once

#include <vector>
#include <cstdint>

namespace core {

class DistanceRings {
public:
    DistanceRings();
    ~DistanceRings() = default;
    
    const std::vector<std::vector<int>>& getRings() const;

private:
    std::vector<std::vector<int>> rings_;
    void initializeRings();
};

} // namespace core