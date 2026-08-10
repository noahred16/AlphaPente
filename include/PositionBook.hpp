#ifndef POSITIONBOOK_HPP
#define POSITIONBOOK_HPP

#include "PNS.hpp"
#include "PenteGame.hpp"
#include "PositionKey.hpp"
#include <iosfwd>
#include <optional>
#include <string>
#include <unordered_map>

// In-RAM store of resolved (WIN/LOSS/DRAW + depth) positions, checkpointed to
// a flat binary file. This is deliberately NOT an embedded KV store
// (RocksDB/LevelDB): PositionKey's exact packing (see PositionKey.hpp) means
// there's no hash-collision-safety argument for reaching for one, and at
// ~9 bytes/record even 10^8 resolved positions is under 1GB, comfortably
// within RAM - see the plan doc for the full reasoning. Revisit only if a
// real solve run's live working set turns out not to fit in memory.
class PositionBook {
  public:
    struct Entry {
        PNS::Outcome outcome = PNS::Outcome::UNKNOWN;
        uint16_t depth = 0;
    };

    // Inserts or overwrites one resolved position.
    void add(PositionKey key, Entry entry);

    // Convenience: pulls every resolved position out of a PNS instance (see
    // PNS::exportResolved()) and adds them all. Typical use: after solve()
    // (or periodically during a long-running one, for checkpointing), call
    // this then save() to persist progress.
    void addAll(const PNS &pns);

    // Position is packed/canonicalized exactly like PNS's own nodes, so a
    // lookup works regardless of which physical orientation the caller's
    // board happens to be in.
    std::optional<Entry> lookup(const PenteGame &game) const;
    std::optional<Entry> lookup(PositionKey key) const;

    size_t size() const { return entries_.size(); }

    // Removes every entry whose position is more than maxMoveCount plies
    // past the empty board (moveCount = occupied-cell count + blackCaptures
    // + whiteCaptures, all recoverable from the packed key itself - see
    // PositionKey.hpp). windowSize must be the packing's true window width
    // (PenteGame::maxIdx()-minIdx(), matching PositionKey::unpack()'s own
    // parameter - not necessarily config().boardSize for even boardSize).
    // Returns the number of entries removed. Typical use: a full
    // solveExhaustive() book is complete but large (every reachable
    // position, however deep); trimming to a shallow moveCount keeps only
    // the opening, on the assumption that a live solve() covers whatever's
    // missing at query time - see apps/Solve5x5.cpp's -m flag and
    // wasm/PenteWasm.cpp's live-search fallback.
    size_t trimToMoveCount(int windowSize, int maxMoveCount);

    // Flat binary format: 4-byte magic "PNTB", uint32 version, uint64 count,
    // then `count` fixed-size records (uint64 key, uint8 packed
    // outcome<<6|depth - see PositionBook.cpp's kVersion comment). Fields
    // are written/read individually rather than as a raw struct dump, so
    // on-disk layout doesn't depend on compiler padding.
    // Returns false (and leaves *this unchanged) on any I/O or format error.
    bool save(const std::string &path) const;
    bool load(const std::string &path);

    // Same format as load(), read from an in-memory buffer instead of a
    // file - e.g. WasmGame (wasm/PenteWasm.cpp) fetches the book bytes over
    // HTTP in JS and passes them in directly, rather than relying on
    // Emscripten's --preload-file (which would force every page load to
    // download the whole book up front, regardless of whether the user ever
    // picks the board size it's for).
    bool loadFromMemory(const uint8_t *data, size_t len);

  private:
    bool loadFromStream(std::istream &is);

    std::unordered_map<PositionKey, Entry> entries_;
};

#endif // POSITIONBOOK_HPP
