#include "PositionBook.hpp"
#include <cstring>
#include <fstream>
#include <sstream>

namespace {
constexpr char kMagic[4] = {'P', 'N', 'T', 'B'};
// v1: uint64 key + uint8 outcome + uint16 depth (11 bytes/record). v2: same
// key, but outcome (2 bits, 4 values exactly - PNS::Outcome is
// UNKNOWN/WIN/LOSS/DRAW) and depth (6 bits, 0-63) packed into a single byte
// (9 bytes/record) - real savings at the ~6M-record scale a full small-board
// book reaches (67MB->~55MB before any transfer compression), and 63 is
// comfortably above any reachable depth for PositionKey::kMaxBoardSize<=5
// (5x5's own worst case is ~43 - see PNS.hpp). save() rejects (returns
// false) rather than silently truncating if that bound is ever exceeded.
constexpr uint32_t kVersion = 2;
constexpr int kDepthBits = 6;
constexpr uint16_t kMaxPackedDepth = (1u << kDepthBits) - 1; // 63

template <typename T> void writeRaw(std::ostream &os, const T &v) { os.write(reinterpret_cast<const char *>(&v), sizeof(T)); }
template <typename T> bool readRaw(std::istream &is, T &v) {
    is.read(reinterpret_cast<char *>(&v), sizeof(T));
    return static_cast<bool>(is);
}
} // namespace

void PositionBook::add(PositionKey key, Entry entry) { entries_[key] = entry; }

void PositionBook::addAll(const PNS &pns) {
    for (const auto &record : pns.exportResolved()) {
        add(record.key, Entry{record.outcome, record.depth});
    }
}

std::optional<PositionBook::Entry> PositionBook::lookup(PositionKey key) const {
    auto it = entries_.find(key);
    if (it == entries_.end()) return std::nullopt;
    return it->second;
}

std::optional<PositionBook::Entry> PositionBook::lookup(const PenteGame &game) const {
    int sym = -1;
    return lookup(PositionKey::canonical(game, sym));
}

bool PositionBook::save(const std::string &path) const {
    std::ofstream os(path, std::ios::binary | std::ios::trunc);
    if (!os) return false;

    os.write(kMagic, sizeof(kMagic));
    writeRaw(os, kVersion);
    writeRaw(os, static_cast<uint64_t>(entries_.size()));

    for (const auto &pair : entries_) {
        if (pair.second.depth > kMaxPackedDepth) return false; // see kVersion's comment
        uint8_t packed = static_cast<uint8_t>((static_cast<uint8_t>(pair.second.outcome) << kDepthBits) |
                                               (pair.second.depth & kMaxPackedDepth));
        writeRaw(os, pair.first.bits);
        writeRaw(os, packed);
    }

    return static_cast<bool>(os);
}

bool PositionBook::load(const std::string &path) {
    std::ifstream is(path, std::ios::binary);
    if (!is) return false;
    return loadFromStream(is);
}

bool PositionBook::loadFromMemory(const uint8_t *data, size_t len) {
    // One extra copy (into the istringstream's internal string) - simple and
    // correct, and only ever runs once per page load in the WASM use case
    // this exists for (see wasm/PenteWasm.cpp), not worth optimizing away
    // with a custom streambuf over `data` directly.
    std::istringstream is(std::string(reinterpret_cast<const char *>(data), len), std::ios::binary);
    return loadFromStream(is);
}

bool PositionBook::loadFromStream(std::istream &is) {
    char magic[4];
    is.read(magic, sizeof(magic));
    if (!is || std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) return false;

    uint32_t version = 0;
    if (!readRaw(is, version) || version != kVersion) return false;

    uint64_t count = 0;
    if (!readRaw(is, count)) return false;

    std::unordered_map<PositionKey, Entry> loaded;
    loaded.reserve(count);
    for (uint64_t i = 0; i < count; ++i) {
        uint64_t bits = 0;
        uint8_t packed = 0;
        if (!readRaw(is, bits) || !readRaw(is, packed)) return false;

        uint8_t outcomeRaw = static_cast<uint8_t>(packed >> kDepthBits);
        uint16_t depth = static_cast<uint16_t>(packed & kMaxPackedDepth);
        if (outcomeRaw > static_cast<uint8_t>(PNS::Outcome::DRAW)) return false; // corrupt/unknown enum value

        loaded[PositionKey{bits}] = Entry{static_cast<PNS::Outcome>(outcomeRaw), depth};
    }

    entries_ = std::move(loaded);
    return true;
}
