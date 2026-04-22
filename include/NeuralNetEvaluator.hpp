#ifndef NEURAL_NET_EVALUATOR_HPP
#define NEURAL_NET_EVALUATOR_HPP

#include "Evaluator.hpp"
#include "NNInputEncoder.hpp"

#ifdef WITH_TORCH

#include "NNModel.hpp"
#include <string>

// Evaluator backed by a PenteNet. Slots directly into MCTS via the Evaluator interface.
//
// A freshly constructed NeuralNetEvaluator has randomly initialized weights — it will
// produce valid (but meaningless) policy/value outputs immediately, which lets you verify
// the full MCTS+NN pipeline before any training has happened.
//
// Usage:
//   NeuralNetEvaluator eval;
//   eval.loadWeights("checkpoint.pt");   // optional — skip for random weights
//
//   MCTS::Config mctsConfig;
//   mctsConfig.evaluator = &eval;
//   MCTS mcts(mctsConfig);
class NeuralNetEvaluator : public Evaluator {
  public:
    explicit NeuralNetEvaluator(const PenteNetConfig &cfg = {});

    void loadWeights(const std::string &path);
    void saveWeights(const std::string &path) const;

    // Returns policy (filtered to legal moves, renormalized) + value in one forward pass.
    std::pair<std::vector<std::pair<PenteGame::Move, float>>, float> evaluate(const PenteGame &game) override;
    std::vector<std::pair<PenteGame::Move, float>> evaluatePolicy(const PenteGame &game) override;
    float evaluateValue(const PenteGame &game) override;

    // Direct access for the training loop.
    PenteNet &model() { return model_; }

  private:
    PenteNet model_;

    // Runs one forward pass. Returns softmax policy over all 361 cells + scalar value.
    std::pair<std::vector<float>, float> forward(const PenteGame &game);
};

#endif // WITH_TORCH
#endif // NEURAL_NET_EVALUATOR_HPP
