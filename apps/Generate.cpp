#ifdef WITH_TORCH

#include "Evaluator.hpp"
#include "NNModel.hpp"
#include "PenteGame.hpp"
#include "SelfPlay.hpp"
#include "TrainCommon.hpp"
#include <iomanip>
#include <iostream>
#include <random>
#include <torch/torch.h>
#include <unistd.h>

int main(int argc, char *argv[]) {
    std::string gameFlag = "pente";
    std::string evalFlag = "auto";  // auto | heuristic | nn
    int gamesPerIter     = 100;
    int mctsSims         = 100;

    int opt;
    while ((opt = getopt(argc, argv, "g:n:s:e:")) != -1) {
        if      (opt == 'g') gameFlag    = optarg;
        else if (opt == 'n') gamesPerIter = std::stoi(optarg);
        else if (opt == 's') mctsSims    = std::stoi(optarg);
        else if (opt == 'e') evalFlag    = optarg;
        else {
            std::cerr << "Usage: generate [-g game] [-n games] [-s sims] [-e auto|heuristic|nn]\n";
            return 1;
        }
    }

    const std::string ckptDir    = std::string(PROJECT_ROOT) + "/checkpoints/" + gameFlag;
    const std::string bestPath   = ckptDir + "/best_model.pt";
    const std::string bufferPath = ckptDir + "/buffer.pt";

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
              << "  games    : " << gamesPerIter  << "\n"
              << "  sims     : " << mctsSims       << "\n"
              << "  evaluator: " << (useHeuristic ? "heuristic" : "nn") << "\n\n";

    SelfPlayConfig spConfig;
    spConfig.simulations      = mctsSims;
    spConfig.explorationC     = 3.0f;
    spConfig.dirichletAlpha   = 0.3f;
    spConfig.dirichletEpsilon = 0.5f;
    spConfig.numWorkerThreads = 6;
    spConfig.numEvalThreads   = useHeuristic ? 0 : 1;  // heuristic uses inline mode

    std::unique_ptr<Evaluator> evalPtr = useHeuristic
        ? std::unique_ptr<Evaluator>(std::make_unique<HeuristicEvaluator>())
        : std::unique_ptr<Evaluator>(std::make_unique<NNEvaluator>(bestPath));
    Evaluator &eval = *evalPtr;

    std::mt19937 rng(std::random_device{}());
    std::vector<torch::Tensor> allPlanes, allCaptures, allPolicies, allValues;
    int totalPositions = 0, bWins = 0, wWins = 0, draws = 0;

    std::cout << "── Self-play (" << gamesPerIter << " games) ──────────────────────────\n";

    for (int g = 0; g < gamesPerIter; g++) {
        auto examples = runGame(eval, gameConfig, spConfig, rng);

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

        if ((g + 1) % 10 == 0 || g + 1 == gamesPerIter)
            std::cout << "  " << std::setw(3) << (g + 1) << "/" << gamesPerIter
                      << "  pos: " << std::setw(6) << totalPositions
                      << "  B/W/D: " << bWins << "/" << wWins << "/" << draws << "\n";
    }

    auto newStates   = torch::stack(allPlanes,   0);
    auto newCaptures = torch::stack(allCaptures, 0);
    auto newPolicies = torch::stack(allPolicies, 0);
    auto newValues   = torch::stack(allValues,   0).unsqueeze(1);

    std::cout << "\n── Buffer ───────────────────────────────────────────────────────\n";
    auto buf = loadBuffer(bufferPath);
    std::cout << "  loaded : " << buf.size() << " positions\n";
    buf = mergeAndTrim(buf, newStates, newCaptures, newPolicies, newValues);
    std::cout << "  updated: " << buf.size() << " positions\n";
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
