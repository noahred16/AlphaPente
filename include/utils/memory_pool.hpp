#pragma once

#include <vector>
#include <cstddef>

namespace utils {

template<typename T>
class MemoryPool {
public:
    MemoryPool(size_t initial_size = 1000);
    ~MemoryPool();
    
    T* allocate();
    void deallocate(T* ptr);
    void clear();

private:
    std::vector<T*> free_list_;
    std::vector<std::vector<T>> blocks_;
    size_t block_size_;
    
    void allocateNewBlock();
};

} // namespace utils