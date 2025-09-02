#pragma once

#include <cstdint>

namespace core {

class Bitboard {
public:
    Bitboard();
    ~Bitboard() = default;

private:
    uint64_t board_;
};

} // namespace core