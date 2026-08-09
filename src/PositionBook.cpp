#include "PositionBook.hpp"
#include <cstring>
#include <fstream>

namespace {
constexpr char kMagic[4] = {'P', 'N', 'T', 'B'};
constexpr uint32_t kVersion = 1;

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
        writeRaw(os, pair.first.bits);
        writeRaw(os, static_cast<uint8_t>(pair.second.outcome));
        writeRaw(os, pair.second.depth);
    }

    return static_cast<bool>(os);
}

bool PositionBook::load(const std::string &path) {
    std::ifstream is(path, std::ios::binary);
    if (!is) return false;

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
        uint8_t outcomeRaw = 0;
        uint16_t depth = 0;
        if (!readRaw(is, bits) || !readRaw(is, outcomeRaw) || !readRaw(is, depth)) return false;
        if (outcomeRaw > static_cast<uint8_t>(PNS::Outcome::DRAW)) return false; // corrupt/unknown enum value

        loaded[PositionKey{bits}] = Entry{static_cast<PNS::Outcome>(outcomeRaw), depth};
    }

    entries_ = std::move(loaded);
    return true;
}
