#include "GameUtils.hpp"
#include "PNS.hpp"
#include "PenteGame.hpp"
#include "PositionBook.hpp"
#ifdef WITH_ROCKSDB
#include "PNSRocks.hpp"
#endif
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <sys/resource.h>
#include <unistd.h>

// mid()'s recursion crashed a real long run (SIGSEGV) once df-pn committed
// deep into one narrow line: PenteGame is 8128 bytes (mostly its embedded
// mt19937) and is passed by value per recursion level, so a default ~8MB
// thread stack only has headroom for a few hundred levels - see
// PNS::Config::maxRecursionDepth's comment for the full explanation. Raise
// this process's stack limit (falls back to whatever the OS actually grants)
// and size maxRecursionDepth to match, rather than relying on PNS's
// conservative small-stack-safe default.
uint64_t raiseStackLimitAndPickDepthBudget() {
    constexpr rlim_t kDesiredStackBytes = 1ULL << 30; // 1 GB
    struct rlimit lim;
    if (getrlimit(RLIMIT_STACK, &lim) != 0) {
        std::cerr << "Warning: getrlimit(RLIMIT_STACK) failed; using PNS's small-stack-safe default depth budget.\n";
        return 300;
    }
    rlim_t target = (lim.rlim_max == RLIM_INFINITY) ? kDesiredStackBytes : std::min(kDesiredStackBytes, lim.rlim_max);
    if (target > lim.rlim_cur) {
        lim.rlim_cur = target;
        if (setrlimit(RLIMIT_STACK, &lim) != 0) {
            std::cerr << "Warning: setrlimit(RLIMIT_STACK) failed; using PNS's small-stack-safe default depth budget.\n";
            return 300;
        }
    }
    std::cout << "Stack limit: " << (target / (1024 * 1024)) << " MB\n";
    // ~20KB/level budgeted (empirically ~2x sizeof(PenteGame) plus overhead
    // for the game param + local childGame coexisting), leaving real margin
    // under what the raised stack can actually support.
    return static_cast<uint64_t>(target) / 20000;
}

// Weak-solve driver: runs df-pn (PNS) from a given position (default: the
// empty board) toward a proven WIN/LOSS/DRAW root, within a time/node budget,
// and saves whatever got resolved to a PositionBook file.
//
// Long runs can be split across sessions via -c/-C (periodically, and always
// at exit, write the ENTIRE in-progress DAG - not just resolved positions -
// to a checkpoint file) and -r (resume df-pn from exactly that state on a
// later run, rather than starting over). This is distinct from -o/-i's
// PositionBook: that format only ever holds resolved positions, for a final
// shippable book; the -c/-r checkpoint is PNS's own working format for
// genuinely resuming an interrupted proof search (see PNS::saveCheckpoint()).
int main(int argc, char *argv[]) {
    int boardSize = 5;
    uint64_t maxNodes = 20'000'000;
    double maxSeconds = 0;
    std::string outPath;
    std::string inPath;
    std::string checkpointOutPath;
    std::string resumePath;
    double checkpointIntervalSeconds = 600;
    bool exhaustive = false;
    int trimToMoveCount = -1; // -1 = don't trim
    std::string rocksDbPath; // -R: use PNSRocks (disk-backed) instead of in-RAM PNS
    uint64_t rocksBlockCacheBytes = 4ULL << 30;
    int opt;
    while ((opt = getopt(argc, argv, "B:N:t:o:i:m:c:C:r:R:M:xh")) != -1) {
        if (opt == 'B') boardSize = std::max(3, std::min(PositionKey::kMaxBoardSize, std::atoi(optarg)));
        else if (opt == 'N') maxNodes = std::strtoull(optarg, nullptr, 10);
        else if (opt == 't') maxSeconds = std::atof(optarg);
        else if (opt == 'o') outPath = optarg;
        else if (opt == 'i') inPath = optarg;
        else if (opt == 'x') exhaustive = true;
        else if (opt == 'm') trimToMoveCount = std::atoi(optarg);
        else if (opt == 'c') checkpointOutPath = optarg;
        else if (opt == 'C') checkpointIntervalSeconds = std::atof(optarg);
        else if (opt == 'r') resumePath = optarg;
        else if (opt == 'R') rocksDbPath = optarg;
        else if (opt == 'M') rocksBlockCacheBytes = std::strtoull(optarg, nullptr, 10);
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
                "  -x              Exhaustive: resolve every reachable position (a full\n"
                "                  book), not just the minimal subset df-pn needs to prove\n"
                "                  the root - can visit substantially more nodes\n"
                "  -m <moveCount>  Trim the book to only positions within moveCount plies\n"
                "                  of the root before saving - a live solve() covers\n"
                "                  whatever's past that at query time (cheap - see the\n"
                "                  project's calibration notes). Only useful with -x/-o.\n"
                "  -o <path>       Save resolved positions to this PositionBook file\n"
                "  -i <path>       Load a PositionBook first; skip solving if the exact\n"
                "                  root is already resolved there\n"
                "  -c <path>       Periodically (and always at exit) checkpoint the entire\n"
                "                  in-progress proof DAG to this file, for -r to resume later\n"
                "  -C <seconds>    Checkpoint interval (default: 600)\n"
                "  -r <path>       Resume df-pn from a checkpoint written by -c, instead of\n"
                "                  starting over (the move string must reproduce the exact\n"
                "                  root position the checkpoint was written from)\n"
#ifdef WITH_ROCKSDB
                "  -R <dir>        Disk-backed mode: store every node in a RocksDB database\n"
                "                  at this path instead of the in-RAM PNS. Ignores -o/-i/-c/\n"
                "                  -r/-x/-m (PNSRocks doesn't support them yet - see\n"
                "                  PNSRocks.hpp). Reopening the same path resumes\n"
                "                  automatically - no separate checkpoint needed.\n"
                "  -M <bytes>      RocksDB block cache size for -R (default: 4294967296 = 4GB)\n"
#endif
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

    if (!rocksDbPath.empty()) {
#ifdef WITH_ROCKSDB
        PNSRocks::Config rocksConfig;
        rocksConfig.dbPath = rocksDbPath;
        rocksConfig.maxNodes = maxNodes;
        rocksConfig.maxSeconds = maxSeconds;
        rocksConfig.maxRecursionDepth = static_cast<int>(std::min<uint64_t>(
            raiseStackLimitAndPickDepthBudget(), static_cast<uint64_t>(INT32_MAX)));
        rocksConfig.blockCacheBytes = rocksBlockCacheBytes;
        std::cout << "Recursion depth budget: " << rocksConfig.maxRecursionDepth << "\n";
        std::cout << "Disk-backed mode: db=" << rocksDbPath << " blockCache=" << (rocksBlockCacheBytes >> 20)
                  << "MB (reopening this path later resumes automatically)\n";
        PNSRocks pns(rocksConfig);

        std::cout << "Solving (disk-backed, maxNodes budget this run=" << maxNodes << ", maxSeconds=" << maxSeconds
                   << ")...\n"
                   << std::flush;
        auto t0 = std::chrono::steady_clock::now();
        bool solved = pns.solve(game);
        double wallSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        pns.printProofStats();
        std::cout << "wall=" << wallSeconds << "s midCalls=" << pns.getStats().midCalls
                  << " nodesCreated=" << pns.getStats().nodesCreated << " dbGets=" << pns.getStats().dbGets
                  << " dbPuts=" << pns.getStats().dbPuts
                  << " getsPerSec=" << (wallSeconds > 0 ? pns.getStats().dbGets / wallSeconds : 0.0) << "\n";
        if (!solved) {
            std::cout << "Not resolved within this run's node budget - DB left in a valid, resumable state; "
                          "rerun with the same -R path to continue.\n";
        }
        return 0;
#else
        std::cerr << "This build was compiled without RocksDB (-R unavailable) - see CMakeLists.txt's "
                      "RocksDB_FOUND guard.\n";
        return 1;
#endif
    }

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
    pnsConfig.maxRecursionDepth = static_cast<int>(std::min<uint64_t>(raiseStackLimitAndPickDepthBudget(),
                                                                       static_cast<uint64_t>(INT32_MAX)));
    if (!checkpointOutPath.empty()) {
        pnsConfig.checkpointPath = checkpointOutPath;
        pnsConfig.checkpointIntervalSeconds = checkpointIntervalSeconds;
        std::cout << "Checkpointing to " << checkpointOutPath << " every " << checkpointIntervalSeconds
                  << "s (and at exit)\n";
    }
    std::cout << "Recursion depth budget: " << pnsConfig.maxRecursionDepth << "\n";
    PNS pns(pnsConfig);

    if (!resumePath.empty()) {
        if (pns.loadCheckpoint(resumePath, game)) {
            std::cout << "Resumed from checkpoint: " << resumePath << " (" << pns.getNodeCount() << " nodes)\n";
        } else {
            std::cerr << "Could not load checkpoint " << resumePath
                       << " (missing, corrupt, or root-player mismatch with the move string above) -"
                          " solving from scratch.\n";
        }
    }

    std::cout << "Solving (" << (exhaustive ? "exhaustive" : "proof-driven") << ", maxNodes=" << maxNodes
              << ", maxSeconds=" << maxSeconds << ")...\n"
              << std::flush;
    auto t0 = std::chrono::steady_clock::now();
    bool solved = exhaustive ? pns.solveExhaustive(game) : pns.solve(game);
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

    if (trimToMoveCount >= 0) {
        // book's true window width matches PenteGame's own (maxIdx()-minIdx()),
        // not necessarily boardSize for even sizes - see PositionKey.hpp.
        int windowSize = game.maxIdx() - game.minIdx();
        size_t removed = book.trimToMoveCount(windowSize, trimToMoveCount);
        std::cout << "Trimmed to moveCount<=" << trimToMoveCount << ": removed " << removed
                  << " positions, " << book.size() << " remain\n";
    }

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
