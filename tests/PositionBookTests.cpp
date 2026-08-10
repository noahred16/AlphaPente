#include "PNS.hpp"
#include "PositionBook.hpp"
#include "PenteGame.hpp"
#include "doctest.h"
#include <cstdio>
#include <fstream>
#include <vector>

namespace {
// Solves the same 3x3 gomoku scenario PNSTests.cpp uses, so this file can
// exercise PositionBook against a real (small, fast) resolved DAG rather than
// hand-built entries alone.
PenteGame solved3x3(PNS &pns) {
    PenteGame::Config config = PenteGame::Config::gomoku();
    config.boardSize = 3;
    PenteGame game(config);
    game.reset();
    game.makeMove(9, 9);
    REQUIRE(pns.solve(game));
    return game;
}
} // namespace

TEST_CASE("PositionBook addAll/lookup round-trips a solved PNS DAG") {
    PNS pns;
    PenteGame game = solved3x3(pns);

    PositionBook book;
    book.addAll(pns);

    // exportResolved() only carries resolved nodes - df-pn's DAG can also
    // contain materialized-but-still-UNKNOWN nodes it touched without needing
    // to fully resolve, so the book is generally smaller than the raw node count.
    CHECK(book.size() > 0);
    CHECK(book.size() <= pns.getNodeCount());

    auto entry = book.lookup(game);
    REQUIRE(entry.has_value());
    CHECK(entry->outcome == pns.getRootOutcome());
    CHECK(entry->depth == pns.getRootDepth());
}

TEST_CASE("PositionBook lookup misses an unvisited position") {
    PositionBook book;
    PenteGame::Config config = PenteGame::Config::gomoku();
    config.boardSize = 5;
    PenteGame game(config);
    game.reset();

    CHECK_FALSE(book.lookup(game).has_value());
}

TEST_CASE("PositionBook save/load round-trips to disk") {
    PNS pns;
    PenteGame game = solved3x3(pns);

    PositionBook book;
    book.addAll(pns);

    const std::string path = "/tmp/positionbook_test_roundtrip.bin";
    REQUIRE(book.save(path));

    PositionBook loaded;
    REQUIRE(loaded.load(path));
    CHECK(loaded.size() == book.size());

    auto entry = loaded.lookup(game);
    REQUIRE(entry.has_value());
    CHECK(entry->outcome == pns.getRootOutcome());
    CHECK(entry->depth == pns.getRootDepth());

    std::remove(path.c_str());
}

TEST_CASE("PositionBook load rejects a missing file") {
    PositionBook book;
    CHECK_FALSE(book.load("/tmp/positionbook_test_does_not_exist.bin"));
}

// loadFromMemory() exists specifically for WasmGame (wasm/PenteWasm.cpp),
// which fetches the book's bytes over HTTP in JS rather than reading a file
// - this checks it parses identically to load(path) given the exact same
// bytes.
TEST_CASE("PositionBook loadFromMemory matches load(path) on the same bytes") {
    PNS pns;
    PenteGame game = solved3x3(pns);

    PositionBook book;
    book.addAll(pns);

    const std::string path = "/tmp/positionbook_test_frommemory.bin";
    REQUIRE(book.save(path));

    std::ifstream is(path, std::ios::binary);
    REQUIRE(is);
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(is)), std::istreambuf_iterator<char>());
    std::remove(path.c_str());

    PositionBook loaded;
    REQUIRE(loaded.loadFromMemory(bytes.data(), bytes.size()));
    CHECK(loaded.size() == book.size());

    auto entry = loaded.lookup(game);
    REQUIRE(entry.has_value());
    CHECK(entry->outcome == pns.getRootOutcome());
    CHECK(entry->depth == pns.getRootDepth());
}

// trimToMoveCount() exists for apps/Solve5x5.cpp's -m flag (keep only the
// opening, rely on a live solve() fallback for anything deeper - see
// wasm/PenteWasm.cpp). Uses solveExhaustive() (not the solved3x3() helper
// above, which only calls solve()) so every reachable position is actually
// present to check the trim against, not just the minimal proof subset.
TEST_CASE("PositionBook trimToMoveCount removes only positions past the cutoff") {
    PenteGame::Config config = PenteGame::Config::gomoku();
    config.boardSize = 3;
    PenteGame game(config);
    game.reset();

    PNS pns;
    REQUIRE(pns.solveExhaustive(game));

    PositionBook book;
    book.addAll(pns);
    const size_t fullSize = book.size();

    const int cutoff = 3;
    size_t removed = book.trimToMoveCount(/*windowSize=*/3, cutoff);

    REQUIRE(removed > 0);
    CHECK(book.size() == fullSize - removed);

    // Deeper position (forced center, then two more moves = moveCount 3) is
    // right at the cutoff and must survive; going one further must not.
    PenteGame atCutoff(config);
    atCutoff.reset();
    atCutoff.makeMove(9, 9);
    atCutoff.makeMove(8, 8);
    atCutoff.makeMove(10, 10);
    CHECK(atCutoff.getMoveCount() == cutoff);
    CHECK(book.lookup(atCutoff).has_value());

    atCutoff.makeMove(8, 10);
    CHECK(atCutoff.getMoveCount() == cutoff + 1);
    CHECK_FALSE(book.lookup(atCutoff).has_value());
}
