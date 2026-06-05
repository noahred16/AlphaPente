#ifdef WITH_TORCH

#include "NNModel.hpp"
#include "TrainCommon.hpp"
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <torch/torch.h>
#include <unistd.h>

static void trainModel(AlphaNet &model, const ReplayBuffer &buf, int gradientSteps) {
    model->train();
    torch::optim::SGD optimizer(
        model->parameters(),
        torch::optim::SGDOptions(LR).momentum(0.9).weight_decay(WEIGHT_DECAY));

    int64_t n             = buf.size();
    double  totalPolicy   = 0.0;
    double  totalValue    = 0.0;
    int     reportEvery   = std::max(1, gradientSteps / 10);

    std::cout << "  baseline policy loss (uniform): ~5.89\n";

    for (int step = 0; step < gradientSteps; step++) {
        auto idx = torch::randint(0, n, {BATCH_SIZE}, torch::kInt64);

        auto bStates   = buf.states.index_select(0, idx);
        auto bCaptures = buf.captures.index_select(0, idx);
        auto bPolicies = buf.policies.index_select(0, idx);
        auto bValues   = buf.values.index_select(0, idx);

        optimizer.zero_grad();
        auto [logPolicy, valuePred] = model->forward(bStates, bCaptures);

        auto pLoss = -(bPolicies * logPolicy).sum(1).mean();
        auto vLoss = torch::mse_loss(valuePred, bValues);
        auto loss  = pLoss + VALUE_LOSS_WEIGHT * vLoss;

        loss.backward();
        optimizer.step();
        totalPolicy += pLoss.item<double>();
        totalValue  += vLoss.item<double>();

        if ((step + 1) % reportEvery == 0 || step + 1 == gradientSteps) {
            double ap = totalPolicy / (step + 1);
            double av = totalValue  / (step + 1);
            std::cout << "  step " << std::setw(4) << (step + 1) << "/" << gradientSteps
                      << "  policy: " << std::fixed << std::setprecision(4) << ap
                      << "  value: "  << std::setprecision(4) << av
                      << "  total: "  << std::setprecision(4) << (ap + av) << "\n";
        }
    }

    model->eval();
}

int main(int argc, char *argv[]) {
    std::string gameFlag      = "pente";
    int         stepsOverride = 0;  // 0 = auto

    int opt;
    while ((opt = getopt(argc, argv, "g:t:")) != -1) {
        if      (opt == 'g') gameFlag      = optarg;
        else if (opt == 't') stepsOverride = std::stoi(optarg);
        else {
            std::cerr << "Usage: train [-g game] [-t steps]\n";
            return 1;
        }
    }

    const std::string ckptDir    = std::string(PROJECT_ROOT) + "/checkpoints/" + gameFlag;
    const std::string bestPath   = ckptDir + "/best_model.pt";
    const std::string bufferPath = ckptDir + "/buffer.pt";

    auto buf = loadBuffer(bufferPath);

    std::cout << "AlphaPente Train\n"
              << "  game  : " << gameFlag   << "\n"
              << "  buffer: " << buf.size() << " positions\n\n";

    if (buf.size() < MIN_BUFFER_SIZE) {
        std::cout << "Buffer (" << buf.size() << ") < min (" << MIN_BUFFER_SIZE
                  << ") — skipping training.\n";
        return 0;
    }

    int gradientSteps = stepsOverride > 0
        ? stepsOverride
        : std::max(MIN_GRAD_STEPS, (int)(buf.size() / BATCH_SIZE));

    std::cout << "── Training (" << gradientSteps << " steps) ──────────────────────────\n";

    AlphaNet model(AlphaNetImpl::kChannels, AlphaNetImpl::kResBlocks);
    torch::load(model, bestPath);
    trainModel(model, buf, gradientSteps);

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
