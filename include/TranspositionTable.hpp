#ifndef TRANSPOSITION_TABLE_HPP
#define TRANSPOSITION_TABLE_HPP

#include <cstdint>
#include <vector>

class TranspositionTable {
  public:
    enum EntryType : uint8_t { EXACT = 0, LOWER_BOUND = 1, UPPER_BOUND = 2 };

    struct Entry {
        uint64_t key = 0;
        float value = 0.0f;
        EntryType type = EXACT;
        uint8_t depth = 0;
        uint16_t age = 0;
    };

    explicit TranspositionTable(size_t sizeInEntries = 1 << 20) : generation_(0) {
        // Round up to power of two
        size_t sz = 1;
        while (sz < sizeInEntries)
            sz <<= 1;
        table_.resize(sz);
        mask_ = sz - 1;
    }

    const Entry *probe(uint64_t key) const {
        const Entry &e = table_[key & mask_];
        if (e.key == key && e.age != 0)
            return &e;
        return nullptr;
    }

    void store(uint64_t key, float value, EntryType type, uint8_t depth) {
        Entry &e = table_[key & mask_];
        // Replace if: empty, same key, older generation, or shallower
        if (e.age == 0 || e.key == key || e.age < generation_ || e.depth <= depth) {
            e.key = key;
            e.value = value;
            e.type = type;
            e.depth = depth;
            e.age = generation_ > 0 ? generation_ : 1; // age=0 means empty
        }
    }

    void newGeneration() { ++generation_; }

    void clear() {
        generation_ = 0;
        for (auto &e : table_) {
            e = Entry{};
        }
    }

  private:
    std::vector<Entry> table_;
    size_t mask_;
    uint16_t generation_;
};

#endif // TRANSPOSITION_TABLE_HPP
