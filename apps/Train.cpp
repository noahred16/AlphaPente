#include "MCTS.hpp"
#include "NNInputEncoder.hpp"
#include "PenteGame.hpp"
#include <iostream>
#include <vector>

#ifdef WITH_TORCH
#include "NeuralNetEvaluator.hpp"
#include <torch/torch.h>

// One self-play training example.
// state:   NNInputEncoder::TOTAL_SIZE floats
// policy:  PLANE_SIZE floats (visit count distribution over all 361 cells)
// outcome: +1 = first player won, -1 = second player won (from that player's perspective)
struct TrainingExample {
    std::vector<float> state;
    std::vector<float> policy;
    float outcome;
};

// Run one self-play game with MCTS and collect training examples.
std::vector<TrainingExample> selfPlay(NeuralNetEvaluator &eval, const MCTS::Config &mctsConfig) {
    PenteGame game;
    MCTS mcts(mctsConfig);

    std::vector<TrainingExample> examples;

    while (!game.isGameOver()) {
        mcts.search(game);

        // Build policy target from root child visit counts.
        // Each child's visit proportion → probability for that cell.
        std::vector<float> policyTarget(NNInputEncoder::PLANE_SIZE, 0.0f);
        int totalVisits = mcts.getTotalVisits();
        if (totalVisits > 0) {
            // TODO: expose root child visits from MCTS for policy targets.
            // For now, use the best move as a one-hot policy (placeholder).
            PenteGame::Move best = mcts.getBestMove();
            policyTarget[best.y * NNInputEncoder::BOARD_SIZE + best.x] = 1.0f;
        }

        examples.push_back({NNInputEncoder::encode(game), policyTarget, 0.0f});

        PenteGame::Move move = mcts.getBestMove();
        game.makeMove(move.x, move.y);
        mcts.reuseSubtree(move);
    }

    // Fill in outcomes now that we know the winner.
    PenteGame::Player winner = game.getWinner();
    // Assign outcome from each position's current-player perspective.
    // Positions alternate Black/White, so outcome alternates sign.
    for (int i = 0; i < int(examples.size()); ++i) {
        PenteGame::Player mover = (i % 2 == 0) ? PenteGame::BLACK : PenteGame::WHITE;
        if (winner == PenteGame::NONE)
            examples[i].outcome = 0.0f;
        else
            examples[i].outcome = (winner == mover) ? 1.0f : -1.0f;
    }

    return examples;
}

// One gradient update step on a batch of examples.
void trainStep(PenteNet &model, torch::optim::Adam &optimizer,
               const std::vector<TrainingExample> &batch) {
    int B = int(batch.size());

    // Stack inputs
    std::vector<torch::Tensor> inputs, policyTargets, valueTargets;
    for (const auto &ex : batch) {
        inputs.push_back(torch::tensor(ex.state)
                             .view({NNInputEncoder::NUM_PLANES,
                                    NNInputEncoder::BOARD_SIZE,
                                    NNInputEncoder::BOARD_SIZE}));
        policyTargets.push_back(torch::tensor(ex.policy));
        valueTargets.push_back(torch::tensor(ex.outcome));
    }

    auto inputBatch  = torch::stack(inputs);                     // [B, C, H, W]
    auto policyBatch = torch::stack(policyTargets);              // [B, 361]
    auto valueBatch  = torch::stack(valueTargets).view({B, 1});  // [B, 1]

    model->train();
    auto [policyLogits, valuePred] = model->forward(inputBatch);

    // Policy loss: cross-entropy between MCTS visit distribution and predicted logits
    auto policyLoss = torch::nn::functional::cross_entropy(policyLogits, policyBatch);

    // Value loss: MSE between game outcome and predicted value
    auto valueLoss = torch::mse_loss(valuePred, valueBatch);

    auto loss = policyLoss + valueLoss;

    optimizer.zero_grad();
    loss.backward();
    optimizer.step();

    std::cout << "  policy_loss=" << policyLoss.item<float>()
              << "  value_loss="  << valueLoss.item<float>() << "\n";
}

int main(int argc, char *argv[]) {
    PenteNetConfig netCfg;
    netCfg.numResBlocks = 5;  // small network to start
    netCfg.numFilters   = 64;

    NeuralNetEvaluator eval(netCfg);

    MCTS::Config mctsCfg;
    mctsCfg.maxIterations = 200;
    mctsCfg.evaluator     = &eval;

    torch::optim::Adam optimizer(eval.model()->parameters(),
                                 torch::optim::AdamOptions(1e-3));

    int numIterations = 100;
    int gamesPerIter  = 5;
    int batchSize     = 32;

    std::vector<TrainingExample> replayBuffer;

    for (int iter = 0; iter < numIterations; ++iter) {
        std::cout << "Iteration " << iter + 1 << "/" << numIterations << "\n";

        // Self-play
        for (int g = 0; g < gamesPerIter; ++g) {
            auto examples = selfPlay(eval, mctsCfg);
            replayBuffer.insert(replayBuffer.end(), examples.begin(), examples.end());
        }

        // Train on random mini-batches from the buffer
        if (int(replayBuffer.size()) >= batchSize) {
            // Shuffle and take one batch (extend to multiple batches as needed)
            std::shuffle(replayBuffer.begin(), replayBuffer.end(),
                         std::mt19937{std::random_device{}()});
            std::vector<TrainingExample> batch(replayBuffer.begin(),
                                               replayBuffer.begin() + batchSize);
            trainStep(eval.model(), optimizer, batch);
        }

        // Cap buffer size to avoid unbounded growth
        if (int(replayBuffer.size()) > 10000)
            replayBuffer.erase(replayBuffer.begin(),
                               replayBuffer.begin() + int(replayBuffer.size()) - 10000);

        // Save checkpoint every 10 iterations
        if ((iter + 1) % 10 == 0) {
            std::string path = "checkpoint_" + std::to_string(iter + 1) + ".pt";
            eval.saveWeights(path);
            std::cout << "  Saved " << path << "\n";
        }
    }

    return 0;
}

#else

int main() {
    std::cerr << "Train requires LibTorch. Rebuild with -DCMAKE_PREFIX_PATH=/path/to/libtorch\n";
    return 1;
}

#endif // WITH_TORCH
