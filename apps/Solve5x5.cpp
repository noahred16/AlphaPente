#include "GameUtils.hpp"
#include "PNS.hpp"
#include "PenteGame.hpp"
#include "PositionBook.hpp"
#include <chrono>
#include <iostream>
#include <unistd.h>

// Weak-solve driver: runs df-pn (PNS) from a given position (default: the
// empty board) toward a proven WIN/LOSS/DRAW root, within a time/node budget,
// and checkpoints whatever got resolved to a PositionBook file.
//
// Known limitation: this does NOT resume a genuinely interrupted search
// mid-DAG - PNS::solve() only exports its *resolved* nodes (PNS::Record),
// not the full in-progress proof tree, so a later run starts df-pn over from
// scratch even with -i pointing at a prior checkpoint. What -i DOES give you:
// if the exact root position was already fully resolved in a prior run, this
// reports that immediately without re-solving. Real mid-search resume would
// need PNS to accept a pre-seeded table_, which is a real chunk of separate
// work (see the project's solve/5x5 plan, Phase 5) - most of the practical
// value for now comes from solving individual opening replies independently
// (each its own bounded, checkpointable run) rather than a giant single call.
int main(int argc, char *argv[]) {
    int boardSize = 5;
    uint64_t maxNodes = 20'000'000;
    double maxSeconds = 0;
    std::string outPath;
    std::string inPath;
    int opt;
    while ((opt = getopt(argc, argv, "B:N:t:o:i:h")) != -1) {
        if (opt == 'B') boardSize = std::max(3, std::min(PositionKey::kMaxBoardSize, std::atoi(optarg)));
        else if (opt == 'N') maxNodes = std::strtoull(optarg, nullptr, 10);
        else if (opt == 't') maxSeconds = std::atof(optarg);
        else if (opt == 'o') outPath = optarg;
        else if (opt == 'i') inPath = optarg;
        else if (opt == 'h') {
            std::cout <<
                "Usage: solve5x5 [options] [\"move string\"]\n"
                "\n"
                "  \"move string\"   PGN-style moves, e.g. \"1. K10 L9\" (default: empty board)\n"
                "\n"
                "Options:\n"
                "  -B <size>       Board size, 3-" << PositionKey::kMaxBoardSize << " (default: 5)\n"
                "  -N <count>      Max transposition-table nodes (default: 20000000)\n"
                "  -t <seconds>    Wall-clock budget, 0 = unlimited (default: 0)\n"
                "  -o <path>       Save resolved positions to this PositionBook file\n"
                "  -i <path>       Load a PositionBook first; skip solving if the exact\n"
                "                  root is already resolved there (see limitation above)\n"
                "  -h              Show this help\n";
            return 0;
        }
    }

    PenteGame::Config config = PenteGame::Config::pente();
    config.boardSize = boardSize;
    config.tournamentRule = false; // PNS requires this off - see PNS.hpp
    PenteGame game(config);
    game.reset();

    const char *gameDataStr = optind < argc ? argv[optind] : "";
    std::vector<std::string> moves = GameUtils::parseGameString(gameDataStr);
    for (const auto &moveStr : moves) game.makeMove(moveStr.c_str());

    std::cout << "Board: " << boardSize << "x" << boardSize << ", tournamentRule disabled\n";
    std::cout << "Parsed moves: ";
    for (const auto &moveStr : moves) std::cout << moveStr << " ";
    std::cout << "\n";
    GameUtils::printGameState(game);

    PositionBook book;
    if (!inPath.empty()) {
        if (book.load(inPath)) {
            std::cout << "Loaded book: " << inPath << " (" << book.size() << " positions)\n";
            auto existing = book.lookup(game);
            if (existing.has_value()) {
                std::cout << "Root already resolved in loaded book: outcome="
                          << (existing->outcome == PNS::Outcome::WIN    ? "WIN"
                              : existing->outcome == PNS::Outcome::LOSS ? "LOSS"
                                                                        : "DRAW")
                          << " depth=" << existing->depth << "\n";
                return 0;
            }
            std::cout << "Root not yet resolved in loaded book; solving from scratch (see -h limitation).\n";
        } else {
            std::cout << "Could not load book at " << inPath << "; solving from scratch.\n";
        }
    }

    PNS::Config pnsConfig;
    pnsConfig.maxNodes = maxNodes;
    pnsConfig.maxSeconds = maxSeconds;
    PNS pns(pnsConfig);

    std::cout << "Solving (maxNodes=" << maxNodes << ", maxSeconds=" << maxSeconds << ")...\n" << std::flush;
    auto t0 = std::chrono::steady_clock::now();
    bool solved = pns.solve(game);
    double wallSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    pns.printProofStats();
    std::cout << "wall=" << wallSeconds << "s midCalls=" << pns.getStats().midCalls
              << " nodesCreated=" << pns.getStats().nodesCreated
              << " transpositionHits=" << pns.getStats().transpositionHits << "\n";

    if (!solved) {
        std::cout << "Not resolved within budget - DAG left in a valid, checkpointable partial state.\n";
    }

    book.addAll(pns);
    std::cout << "Resolved positions this run: " << pns.exportResolved().size() << "\n";

    if (!outPath.empty()) {
        if (book.save(outPath)) {
            std::cout << "Saved book: " << outPath << " (" << book.size() << " positions)\n";
        } else {
            std::cerr << "Failed to save book to " << outPath << "\n";
            return 1;
        }
    }

    return 0;
}
