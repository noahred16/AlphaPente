#ifdef WITH_TORCH

#include "Evaluator.hpp"
#include "NNModel.hpp"
#include "ParallelMCTS.hpp"
#include "PenteGame.hpp"
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <random>
#include <torch/torch.h>
#include <unistd.h>
#include <vector>

// ── Hyperparameters ───────────────────────────────────────────────────────────

static constexpr int   GAMES_PER_ITER    = 100;
static constexpr int   MCTS_SIMS         = 100;
static constexpr int   TEMP_DROPOFF      = 15;   // moves before switching to greedy
static constexpr int   BUFFER_SIZE       = 100'000;
static constexpr int   MIN_BUFFER_SIZE   = 5'000;
static constexpr int   BATCH_SIZE        = 256;
static constexpr float LR                = 0.01f;
static constexpr float WEIGHT_DECAY      = 1e-4f;
static constexpr float VALUE_LOSS_WEIGHT = 1.0f;

// ── Replay buffer ─────────────────────────────────────────────────────────────

struct ReplayBuffer {
    torch::Tensor states;    // [N, 3, 19, 19]
    torch::Tensor captures;  // [N, 2]
    torch::Tensor policies;  // [N, 361]
    torch::Tensor values;    // [N, 1]
    int64_t size() const { return states.defined() ? states.size(0) : 0; }
};

static ReplayBuffer loadBuffer(const std::string &path) {
    ReplayBuffer buf;
    if (!std::filesystem::exists(path)) return buf;
    try {
        torch::serialize::InputArchive ar;
        ar.load_from(path);
        ar.read("states",   buf.states);
        ar.read("captures", buf.captures);
        ar.read("policies", buf.policies);
        ar.read("values",   buf.values);
    } catch (const std::exception &e) {
        std::cerr << "Warning: failed to load buffer (" << e.what() << ")\n";
    }
    return buf;
}

static void saveBuffer(const ReplayBuffer &buf, const std::string &path) {
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    torch::serialize::OutputArchive ar;
    ar.write("states",   buf.states);
    ar.write("captures", buf.captures);
    ar.write("policies", buf.policies);
    ar.write("values",   buf.values);
    ar.save_to(path);
}

static ReplayBuffer mergeAndTrim(ReplayBuffer existing,
                                  torch::Tensor newStates,
                                  torch::Tensor newCaptures,
                                  torch::Tensor newPolicies,
                                  torch::Tensor newValues) {
    if (!existing.states.defined()) {
        existing = {newStates, newCaptures, newPolicies, newValues};
    } else {
        existing.states   = torch::cat({existing.states,   newStates},   0);
        existing.captures = torch::cat({existing.captures, newCaptures}, 0);
        existing.policies = torch::cat({existing.policies, newPolicies}, 0);
        existing.values   = torch::cat({existing.values,   newValues},   0);
    }
    int64_t n = existing.states.size(0);
    if (n > BUFFER_SIZE) {
        int64_t drop = n - BUFFER_SIZE;
        existing.states   = existing.states.slice(0, drop);
        existing.captures = existing.captures.slice(0, drop);
        existing.policies = existing.policies.slice(0, drop);
        existing.values   = existing.values.slice(0, drop);
    }
    return existing;
}

// ── Self-play ─────────────────────────────────────────────────────────────────

struct Example {
    torch::Tensor planes;
    torch::Tensor captures;
    torch::Tensor policy;
    PenteGame::Player player;
    float value = 0.0f;
};

static std::vector<Example> runGame(Evaluator &eval,
                                     const PenteGame::Config &config,
                                     int sims,
                                     std::mt19937 &rng) {
    PenteGame game(config);
    game.reset();

    ParallelMCTS::Config mctsConfig;
    mctsConfig.maxIterations    = sims;
    mctsConfig.evaluator        = &eval;
    mctsConfig.numWorkerThreads = 4;
    mctsConfig.numEvalThreads   = 1;
    mctsConfig.seed             = static_cast<uint32_t>(rng());
    ParallelMCTS mcts(mctsConfig);

    constexpr int B = PenteGame::BOARD_SIZE;
    std::vector<Example> examples;

    while (!game.isGameOver()) {
        mcts.search(game);

        const auto *root = mcts.getRoot();
        int cap          = static_cast<int>(root->childCapacity);

        // Collect visit counts from root children
        std::vector<int> visits(cap, 0);
        int totalVisits = 0;
        for (int i = 0; i < cap; i++) {
            if (root->children[i])
                visits[i] = root->children[i]->visits.load();
            totalVisits += visits[i];
        }

        // Build flat [361] policy target from visit proportions
        auto policyTensor = torch::zeros({B * B});
        if (totalVisits > 0) {
            auto acc = policyTensor.accessor<float, 1>();
            for (int i = 0; i < cap; i++) {
                if (visits[i] > 0) {
                    const auto &mv = root->moves[i];
                    acc[mv.y * B + mv.x] = (float)visits[i] / (float)totalVisits;
                }
            }
        }

        auto [planes, captures] = NNEvaluator::gameToTensors(game);
        examples.push_back({planes, captures, policyTensor, game.getCurrentPlayer(), 0.0f});

        // Select move: proportional sampling early, greedy after TEMP_DROPOFF
        int chosen = 0;
        if (game.getMoveCount() >= TEMP_DROPOFF || totalVisits == 0) {
            chosen = (int)(std::max_element(visits.begin(), visits.end()) - visits.begin());
        } else {
            std::uniform_int_distribution<int> dist(0, totalVisits - 1);
            int r = dist(rng), cum = 0;
            for (int i = 0; i < cap; i++) {
                cum += visits[i];
                if (r < cum) { chosen = i; break; }
            }
        }

        PenteGame::Move mv = root->moves[chosen];
        mcts.reset();
        game.makeMove(mv.x, mv.y);
    }

    // Fill value targets retroactively from game outcome
    PenteGame::Player winner = game.getWinner();
    for (auto &ex : examples)
        ex.value = (winner == PenteGame::NONE) ? 0.0f
                 : (ex.player == winner)        ? 1.0f
                                                : -1.0f;

    return examples;
}

// ── Training ──────────────────────────────────────────────────────────────────

static void trainModel(AlphaNet &model, const ReplayBuffer &buf, int gradientSteps) {
    model->train();
    torch::optim::SGD optimizer(
        model->parameters(),
        torch::optim::SGDOptions(LR).momentum(0.9).weight_decay(WEIGHT_DECAY));

    int64_t n        = buf.size();
    double totalLoss = 0.0;

    for (int step = 0; step < gradientSteps; step++) {
        auto idx = torch::randint(0, n, {BATCH_SIZE}, torch::kInt64);

        auto bStates   = buf.states.index_select(0, idx);
        auto bCaptures = buf.captures.index_select(0, idx);
        auto bPolicies = buf.policies.index_select(0, idx);
        auto bValues   = buf.values.index_select(0, idx);

        optimizer.zero_grad();
        auto [logPolicy, valuePred] = model->forward(bStates, bCaptures);

        auto policyLoss = -(bPolicies * logPolicy).sum(1).mean();
        auto valueLoss  = torch::mse_loss(valuePred, bValues);
        auto loss       = policyLoss + VALUE_LOSS_WEIGHT * valueLoss;

        loss.backward();
        optimizer.step();
        totalLoss += loss.item<double>();

        if ((step + 1) % 50 == 0 || step + 1 == gradientSteps)
            std::cout << "  step " << std::setw(4) << (step + 1) << "/" << gradientSteps
                      << "  avg loss: " << std::fixed << std::setprecision(4)
                      << totalLoss / (step + 1) << "\n";
    }

    model->eval();
}

// ── Checkpoint helpers ────────────────────────────────────────────────────────

static int nextIterNumber(const std::string &dir) {
    int maxIter = 0;
    if (!std::filesystem::exists(dir)) return 1;
    for (const auto &entry : std::filesystem::directory_iterator(dir)) {
        const std::string stem = entry.path().stem().string();
        if (stem.rfind("model_iter", 0) == 0) {
            try { maxIter = std::max(maxIter, std::stoi(stem.substr(10))); }
            catch (...) {}
        }
    }
    return maxIter + 1;
}

// ── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char *argv[]) {
    std::string gameFlag = "pente";
    int gamesPerIter     = GAMES_PER_ITER;
    int mctsSims         = MCTS_SIMS;

    int opt;
    while ((opt = getopt(argc, argv, "g:n:s:")) != -1) {
        if      (opt == 'g') gameFlag     = optarg;
        else if (opt == 'n') gamesPerIter = std::stoi(optarg);
        else if (opt == 's') mctsSims     = std::stoi(optarg);
    }

    const std::string ckptDir    = std::string(PROJECT_ROOT) + "/checkpoints/" + gameFlag;
    const std::string bestPath   = ckptDir + "/best_model.pt";
    const std::string bufferPath = ckptDir + "/buffer.pt";

    std::cout << "AlphaPente Training\n"
              << "  game  : " << gameFlag    << "\n"
              << "  games : " << gamesPerIter << "\n"
              << "  sims  : " << mctsSims     << "\n\n";

    std::filesystem::create_directories(ckptDir);
    if (!std::filesystem::exists(bestPath)) {
        std::cout << "Initializing new model...\n";
        AlphaNet m(AlphaNetImpl::kChannels, AlphaNetImpl::kResBlocks);
        torch::save(m, bestPath);
        std::cout << "Saved → " << bestPath << "\n\n";
    }

    PenteGame::Config gameConfig =
        (gameFlag == "gomoku")     ? PenteGame::Config::gomoku()     :
        (gameFlag == "keryopente") ? PenteGame::Config::keryoPente() :
                                     PenteGame::Config::pente();

    std::mt19937 rng(std::random_device{}());

    // ── Phase 1: Self-play ────────────────────────────────────────────────────
    std::cout << "── Self-play (" << gamesPerIter << " games) ──────────────────────────\n";

    NNEvaluator eval(bestPath);
    std::vector<torch::Tensor> allPlanes, allCaptures, allPolicies, allValues;
    int totalPositions = 0, bWins = 0, wWins = 0, draws = 0;

    for (int g = 0; g < gamesPerIter; g++) {
        auto examples = runGame(eval, gameConfig, mctsSims, rng);

        if (!examples.empty()) {
            float v0 = examples[0].value;  // Black's first-move value
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
    auto newValues   = torch::stack(allValues,   0).unsqueeze(1);  // [N, 1]

    // ── Phase 2: Buffer ───────────────────────────────────────────────────────
    std::cout << "\n── Buffer ───────────────────────────────────────────────────────\n";
    auto buf = loadBuffer(bufferPath);
    std::cout << "  loaded : " << buf.size() << " positions\n";
    buf = mergeAndTrim(buf, newStates, newCaptures, newPolicies, newValues);
    std::cout << "  updated: " << buf.size() << " positions\n";
    saveBuffer(buf, bufferPath);
    std::cout << "  saved  → " << bufferPath << "\n";

    // ── Phase 3: Training ─────────────────────────────────────────────────────
    if (buf.size() < MIN_BUFFER_SIZE) {
        std::cout << "\nBuffer (" << buf.size() << ") < min (" << MIN_BUFFER_SIZE
                  << ") — skipping training.\n";
        return 0;
    }

    int gradientSteps = (int)(buf.size() / BATCH_SIZE);
    std::cout << "\n── Training (" << gradientSteps << " steps) ──────────────────────────\n";

    AlphaNet model(AlphaNetImpl::kChannels, AlphaNetImpl::kResBlocks);
    torch::load(model, bestPath);
    trainModel(model, buf, gradientSteps);

    // ── Phase 4: Save ─────────────────────────────────────────────────────────
    int iterNum = nextIterNumber(ckptDir);
    char iterSuffix[8];
    std::snprintf(iterSuffix, sizeof(iterSuffix), "%04d", iterNum);
    std::string iterPath = ckptDir + "/model_iter" + iterSuffix + ".pt";

    torch::save(model, iterPath);
    torch::save(model, bestPath);

    std::cout << "\n── Saved ────────────────────────────────────────────────────────\n"
              << "  " << iterPath << "\n"
              << "  " << bestPath << "\n";

    return 0;
}

#else

#include <iostream>
int main() {
    std::cout << "LibTorch not available — build with Torch to enable training.\n";
    return 1;
}

#endif
