#include "PNS.hpp"
#include "PositionBook.hpp"
#include "PenteGame.hpp"
#include "doctest.h"
#include <cstdio>

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
