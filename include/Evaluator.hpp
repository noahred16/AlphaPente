#ifndef EVALUATOR_HPP
#define EVALUATOR_HPP

#include "PenteGame.hpp"
#include <memory>
#include <string>
#include <utility>
#include <vector>

// Abstract interface for position evaluation
class Evaluator {
  public:
    virtual ~Evaluator() = default;

    // returns policy and value together
    virtual std::pair<std::vector<std::pair<PenteGame::Move, float>>, float> evaluate(const PenteGame &game) = 0;

    // returns orderd pairs of moves and their probabilities, sorted by probability. filtered to only legal moves.
    virtual std::vector<std::pair<PenteGame::Move, float>> evaluatePolicy(const PenteGame &game) = 0;

    // Returns expected value from current player's perspective
    // Returns: value in range [-1.0, 1.0] where 1.0 = current player winning
    virtual float evaluateValue(const PenteGame &game) = 0;

    // Batch evaluation — default loops over evaluate(); NNEvaluator overrides with one forward pass
    virtual std::vector<std::pair<std::vector<std::pair<PenteGame::Move, float>>, float>>
        evaluateBatch(const std::vector<PenteGame> &games);

    void setMaxRolloutDepth(int depth) { maxRolloutDepth_ = depth; }

  protected:
    int maxRolloutDepth_ = 200;

    // Random rollout from current position, returns value in same convention as evaluateValue
    float rollout(const PenteGame &game);
};

// Baseline - uniform policy, rollout for value
class UniformEvaluator : public Evaluator {
  public:
    UniformEvaluator() = default;
    ~UniformEvaluator() override = default;
    std::pair<std::vector<std::pair<PenteGame::Move, float>>, float> evaluate(const PenteGame &game) override;
    std::vector<std::pair<PenteGame::Move, float>> evaluatePolicy(const PenteGame &game) override;
    float evaluateValue(const PenteGame &game) override;
};

// Heuristic Evaluator
class HeuristicEvaluator : public Evaluator {
  public:
    HeuristicEvaluator() = default;
    ~HeuristicEvaluator() override = default;
    std::pair<std::vector<std::pair<PenteGame::Move, float>>, float> evaluate(const PenteGame &game) override;
    std::vector<std::pair<PenteGame::Move, float>> evaluatePolicy(const PenteGame &game) override;
    float evaluateValue(const PenteGame &game) override;
};

#ifdef WITH_TORCH
#include "NNModel.hpp"
#include <torch/torch.h>
class NNEvaluator : public Evaluator {
  public:
    // channels/resBlocks must match whatever the loaded checkpoint (or, for
    // the no-path ctor, the model about to be trained) was/will be saved
    // with — weight shapes are fixed at save time, so a mismatch fails to load.
    explicit NNEvaluator(int channels = AlphaNetImpl::kChannels, int resBlocks = AlphaNetImpl::kResBlocks);  // randomly initialized weights (iteration 0)
    explicit NNEvaluator(const std::string &modelPath, int channels = AlphaNetImpl::kChannels, int resBlocks = AlphaNetImpl::kResBlocks);
    ~NNEvaluator() override;
    std::pair<std::vector<std::pair<PenteGame::Move, float>>, float> evaluate(const PenteGame &game) override;
    std::vector<std::pair<PenteGame::Move, float>> evaluatePolicy(const PenteGame &game) override;
    float evaluateValue(const PenteGame &game) override;

    static std::pair<torch::Tensor, torch::Tensor> gameToTensors(const PenteGame &game);

    std::vector<std::pair<std::vector<std::pair<PenteGame::Move, float>>, float>>
        evaluateBatch(const std::vector<PenteGame> &games) override;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
#endif // WITH_TORCH

#endif // EVALUATOR_HPP