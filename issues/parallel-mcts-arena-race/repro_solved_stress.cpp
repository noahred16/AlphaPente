// Standalone repro harness: isolate the "position gets solved mid-search, and
// every worker funnels through the same proven subtree for the rest of the
// run" stress pattern from arena exhaustion. Uses a LARGE (non-limiting)
// arena so the only interesting activity during the run is normal search
// plus minimax solved-status convergence — not the allocator running out of
// room.
//
// Starts from the real-world crash position (same opening as
// repro_arena.cpp / the original web_scrape crash) but appends a few moves
// where White plays uselessly (far corner, no defensive value) while Black
// builds an open four — a guaranteed win in <=2 plies. Under normal (fully
// contested) play the tree can take many millions of iterations to converge
// on a proof, too slow to observe under TSan in a reasonable wall-clock
// window; this handicap makes convergence fast and reliable while keeping
// the rest of the board's real complexity intact.
//
// Prints tree-size progress every second (from a separate polling thread)
// so it's visible whether the run actually reached meaningful search volume
// before being killed/timed out, since TSan instrumentation overhead can be
// severe enough that a short wall-clock window never gets past the opening.
#include "Evaluator.hpp"
#include "GameUtils.hpp"
#include "ParallelMCTS.hpp"
#include "PenteGame.hpp"
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>

int main(int argc, char *argv[]) {
    int numThreads      = argc > 1 ? std::atoi(argv[1]) : 6;
    long long maxIters   = argc > 2 ? std::atoll(argv[2]) : 3000000;
    size_t arenaBytes    = argc > 3 ? static_cast<size_t>(std::atoll(argv[3])) : (2ull * 1024 * 1024 * 1024);

    // Real 12-move crash opening, plus 3 more moves: Black plays K15-K17
    // (an open three, K14/K18 both empty) while White wastes moves in the
    // far corner. An open four was too obvious — the evaluator's policy
    // found and proved it in ~6 visits, too fast to build up real tree
    // contention. An open three needs White to actually respond (and still
    // loses), giving the tree a few more plies — and real node volume —
    // to grow through before converging.
    const char *hardCodedGame =
        "1.  H8 H7  2.  H6 J7  3.  G7 J9  4.  J5 K4  5.  F8 E9  6.  G8 J8  "
        "7.  J10 H9  8.  G9 G6  9.  F5 G10  10.  K7 F6  11.  K9 E10  12.  E8 D8  "
        "13. K15 A19  14. K16 B19  15. K17 A18 ";

    std::vector<std::string> moves = GameUtils::parseGameString(hardCodedGame);

    PenteGame::Config renjuConfig = PenteGame::Config::renju();
    renjuConfig.numOffsets = 16;
    PenteGame game(renjuConfig);
    game.reset();
    for (const auto &moveStr : moves) game.makeMove(moveStr.c_str());

    HeuristicEvaluator heuristicEvaluator;

    ParallelMCTS::Config config;
    config.maxIterations = static_cast<int>(maxIters);
    config.explorationConstant = 1.414;
    config.numWorkerThreads = numThreads;
    config.numEvalThreads = 0;
    config.arenaSize = arenaBytes;
    config.evaluator = &heuristicEvaluator;

    std::cout << "arena=" << arenaBytes << " bytes, threads=" << numThreads
              << ", maxIterations=" << maxIters << std::endl;

    ParallelMCTS mcts(config);

    std::atomic<bool> done{false};
    std::thread progress([&] {
        while (!done.load()) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            std::cout << "  progress: nodes=" << mcts.getTreeSize()
                      << " visits=" << mcts.getTotalVisits() << std::endl;
        }
    });

    PenteGame::Move best = mcts.search(game);
    done.store(true);
    progress.join();

    std::cout << "DONE best=(" << best.x << "," << best.y << ") nodes=" << mcts.getTreeSize()
              << " visits=" << mcts.getTotalVisits() << std::endl;
    return 0;
}
