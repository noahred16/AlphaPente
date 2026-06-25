#ifdef WITH_TORCH

#include "Evaluator.hpp"
#include "NNModel.hpp"
#include "PenteGame.hpp"
#include "SelfPlay.hpp"
#include "TrainCommon.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <torch/torch.h>
#include <unistd.h>

int main(int argc, char *argv[]) {
    std::string gameFlag = "pente";
    std::string evalFlag = "auto";  // auto | heuristic | nn
    int gamesPerIter     = 100;
    int mctsSims         = 20000;
    bool bootstrap       = false;
    int tailMoves        = 0;     // 0 = all moves
    bool augment         = false;

    auto usage = [&](std::ostream &out) {
        out <<
            "Usage: generate [-g game] [-n games] [-s sims] [-e evaluator] [-b] [-t tail] [-a]\n"
            "\n"
            "Options:\n"
            "  -g  game: pente | gomoku | keryopente      (default: " << gameFlag << ")\n"
            "  -n  number of self-play games               (default: " << gamesPerIter << ")\n"
            "  -s  MCTS simulations per move               (default: " << mctsSims << ")\n"
            "  -e  evaluator: auto | heuristic | nn        (default: auto)\n"
            "  -b  bootstrap mode — use heuristic, write to bootstrap.pt\n"
            "  -t  keep only the last N moves per game     (default: all)\n"
            "  -a  augment positions with 8 board symmetries\n"
            "\n"
            "Examples:\n"
            "  # as called by train_loop.sh during self-play training\n"
            "  ./generate -g pente -n 50 -s 400 -a\n"
            "\n"
            "  # ad hoc: bootstrap the buffer with heuristic-guided play\n"
            "  ./generate -g pente -n 200 -s 200 -b\n";
    };

    int opt;
    while ((opt = getopt(argc, argv, "g:n:s:e:bt:ah")) != -1) {
        if      (opt == 'g') gameFlag     = optarg;
        else if (opt == 'n') gamesPerIter = std::stoi(optarg);
        else if (opt == 's') mctsSims     = std::stoi(optarg);
        else if (opt == 'e') evalFlag     = optarg;
        else if (opt == 'b') bootstrap    = true;
        else if (opt == 't') tailMoves    = std::stoi(optarg);
        else if (opt == 'a') augment      = true;
        else if (opt == 'h') { usage(std::cout); return 0; }
        else                 { usage(std::cerr); return 1; }
    }

    if (bootstrap) evalFlag = "heuristic";

    const std::string ckptDir    = std::string(PROJECT_ROOT) + "/checkpoints/" + gameFlag;
    const std::string bestPath   = ckptDir + "/best_model.pt";
    const std::string bufferPath = bootstrap ? ckptDir + "/bootstrap.pt"
                                             : ckptDir + "/buffer.pt";

    std::filesystem::create_directories(ckptDir);
    if (!std::filesystem::exists(bestPath)) {
        std::cout << "Initializing new model...\n";
        AlphaNet m(AlphaNetImpl::kChannels, AlphaNetImpl::kResBlocks);
        torch::save(m, bestPath);
        std::cout << "Saved → " << bestPath << "\n\n";
    }

    bool useHeuristic;
    if      (evalFlag == "heuristic") useHeuristic = true;
    else if (evalFlag == "nn")        useHeuristic = false;
    else                              useHeuristic = (nextIterNumber(ckptDir) == 1);

    PenteGame::Config gameConfig =
        (gameFlag == "gomoku")     ? PenteGame::Config::gomoku()     :
        (gameFlag == "keryopente") ? PenteGame::Config::keryoPente() :
                                     PenteGame::Config::pente();

    std::cout << "AlphaPente Generate\n"
              << "  game     : " << gameFlag      << "\n"
              << "  mode     : " << (bootstrap ? "bootstrap" : "selfplay") << "\n"
              << "  games    : " << gamesPerIter  << "\n"
              << "  sims     : " << mctsSims       << "\n"
              << "  evaluator: " << (useHeuristic ? "heuristic" : "nn") << "\n"
              << "  tail     : " << (tailMoves > 0 ? std::to_string(tailMoves) + " moves" : "all") << "\n"
              << "  augment  : " << (augment ? "yes (8x)" : "no") << "\n\n";

    if (!useHeuristic)
        torch::set_num_threads(1);  // 1w/6e: each eval thread runs its own forward pass

    SelfPlayConfig spConfig;
    spConfig.simulations      = mctsSims;
    spConfig.explorationC     = 3.0f;
    spConfig.dirichletAlpha   = 0.3f;
    spConfig.dirichletEpsilon = 0.25f;
    spConfig.numWorkerThreads = useHeuristic ? GameUtils::numThreadsFromEnv() : 1;
    spConfig.numEvalThreads   = useHeuristic ? 0 : 6;

    std::unique_ptr<Evaluator> evalPtr = useHeuristic
        ? std::unique_ptr<Evaluator>(std::make_unique<HeuristicEvaluator>())
        : std::unique_ptr<Evaluator>(std::make_unique<NNEvaluator>(bestPath));
    Evaluator &eval = *evalPtr;

    std::mt19937 rng(std::random_device{}());
    std::vector<torch::Tensor> allPlanes, allCaptures, allPolicies, allValues;
    int totalPositions = 0, bWins = 0, wWins = 0, draws = 0;
    int lastGameMoves = 0;

    std::cout << "── Self-play (" << gamesPerIter << " games) ──────────────────────────\n";

    auto t0 = std::chrono::steady_clock::now();

    for (int g = 0; g < gamesPerIter; g++) {
        auto tGame = std::chrono::steady_clock::now();
        auto examples = runGame(eval, gameConfig, spConfig, rng);

        lastGameMoves = (int)examples.size();
        if (tailMoves > 0 && (int)examples.size() > tailMoves)
            examples.erase(examples.begin(), examples.end() - tailMoves);

        if (!examples.empty()) {
            float v0 = examples[0].value;
            if      (v0 >  0.5f) bWins++;
            else if (v0 < -0.5f) wWins++;
            else                 draws++;
        }

        for (auto &ex : examples) {
            allPlanes.push_back(ex.planes);
            allCaptures.push_back(ex.captures);
            allPolicies.push_back(ex.policy);
            allValues.push_back(torch::tensor(ex.value));
            totalPositions++;
        }

        if ((g + 1) % 10 == 0 || g + 1 == gamesPerIter) {
            double gameSecs = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - tGame).count();
            double elapsed  = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            double avgSecs  = elapsed / (g + 1);
            std::cout << "  " << std::setw(3) << (g + 1) << "/" << gamesPerIter
                      << "  pos: "  << std::setw(6) << totalPositions
                      << "  B/W/D: " << bWins << "/" << wWins << "/" << draws
                      << "  moves: " << std::setw(3) << lastGameMoves
                      << "  game: " << std::fixed << std::setprecision(1) << gameSecs << "s"
                      << "  avg: "  << std::setprecision(1) << avgSecs << "s\n";
        }
    }

    double totalSecs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    int preAugPositions = totalPositions;
    std::cout << "\n  total: " << std::fixed << std::setprecision(1) << totalSecs << "s"
              << "  avg/game: " << std::setprecision(1) << totalSecs / gamesPerIter << "s"
              << "  avg/pos: "  << std::setprecision(3) << (totalPositions > 0 ? totalSecs / totalPositions : 0.0) << "s\n";

    if (augment) {
        std::vector<torch::Tensor> augPlanes, augCaptures, augPolicies, augValues;
        augPlanes.reserve(allPlanes.size() * 8);
        augCaptures.reserve(allCaptures.size() * 8);
        augPolicies.reserve(allPolicies.size() * 8);
        augValues.reserve(allValues.size() * 8);

        constexpr int B = PenteGame::BOARD_SIZE;
        for (int i = 0; i < (int)allPlanes.size(); i++) {
            auto pol2d = allPolicies[i].view({B, B});
            for (int flip = 0; flip < 2; flip++) {
                auto p   = flip ? torch::flip(allPlanes[i], {2})   : allPlanes[i];
                auto pol = flip ? torch::flip(pol2d,        {1})   : pol2d;
                for (int rot = 0; rot < 4; rot++) {
                    augPlanes.push_back(rot > 0 ? torch::rot90(p,   rot, {1, 2}) : p);
                    augPolicies.push_back((rot > 0 ? torch::rot90(pol, rot, {0, 1}) : pol).flatten());
                    augCaptures.push_back(allCaptures[i]);
                    augValues.push_back(allValues[i]);
                }
            }
        }
        allPlanes   = std::move(augPlanes);
        allCaptures = std::move(augCaptures);
        allPolicies = std::move(augPolicies);
        allValues   = std::move(augValues);
        totalPositions = (int)allPlanes.size();
        std::cout << "  augmented: " << preAugPositions << " → " << totalPositions
                  << " positions (8x)\n";
    }

    auto newStates   = torch::stack(allPlanes,   0);
    auto newCaptures = torch::stack(allCaptures, 0);
    auto newPolicies = torch::stack(allPolicies, 0);
    auto newValues   = torch::stack(allValues,   0).unsqueeze(1);

    std::cout << "\n── " << (bootstrap ? "Bootstrap" : "Buffer")
              << " ───────────────────────────────────────────────────────\n";
    auto buf = loadBuffer(bufferPath);
    std::cout << "  existing: " << buf.size() << " positions\n"
              << "  new     : " << totalPositions << " positions\n";
    if (bootstrap) {
        buf = mergeAndTrim(buf, newStates, newCaptures, newPolicies, newValues, /*maxSize=*/0);
    } else {
        buf = mergeAndTrim(buf, newStates, newCaptures, newPolicies, newValues);
    }
    std::cout << "  total   : " << buf.size() << " positions\n";
    saveBuffer(buf, bufferPath);
    std::cout << "  saved  → " << bufferPath << "\n";

    return 0;
}

#else

#include <iostream>
int main() {
    std::cout << "LibTorch not available — build with Torch to enable generate.\n";
    return 1;
}

#endif
