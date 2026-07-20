#ifdef WITH_TORCH

#include "NNModel.hpp"
#include "TrainCommon.hpp"
#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <torch/torch.h>
#include <unistd.h>

// Fixed seed: baseline and candidate must be scored on the identical held-out
// split within one run for the promotion gate to be a fair comparison.
static std::pair<ReplayBuffer, ReplayBuffer> splitBuffer(const ReplayBuffer &buf, double valFraction) {
    int64_t n    = buf.size();
    int64_t valN = std::max<int64_t>(1, (int64_t)(n * valFraction));

    std::vector<int64_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::mt19937_64 rng(1234);
    std::shuffle(idx.begin(), idx.end(), rng);

    auto idxTensor = torch::from_blob(idx.data(), {n}, torch::kInt64).clone();
    auto valIdx    = idxTensor.slice(0, 0, valN);
    auto trainIdx  = idxTensor.slice(0, valN, n);

    auto sel = [](const torch::Tensor &t, const torch::Tensor &i) { return t.index_select(0, i); };
    ReplayBuffer trainBuf{sel(buf.states, trainIdx), sel(buf.captures, trainIdx),
                          sel(buf.policies, trainIdx), sel(buf.values, trainIdx)};
    ReplayBuffer valBuf{sel(buf.states, valIdx), sel(buf.captures, valIdx),
                        sel(buf.policies, valIdx), sel(buf.values, valIdx)};
    return {trainBuf, valBuf};
}

static double computeValLoss(AlphaNet &model, const ReplayBuffer &val, torch::Device device) {
    torch::NoGradGuard noGrad;
    model->to(device);
    model->eval();

    int64_t n = val.size();
    if (n == 0) return 0.0;

    double  totalLoss  = 0.0;
    int64_t numBatches = 0;
    for (int64_t start = 0; start < n; start += BATCH_SIZE) {
        int64_t end = std::min<int64_t>(start + BATCH_SIZE, n);

        auto captures = val.captures.slice(0, start, end).to(device);
        auto states   = decodeStates(val.states.slice(0, start, end).to(device), captures);
        auto policies = val.policies.slice(0, start, end).to(device, torch::kFloat);
        auto values   = val.values.slice(0, start, end).to(device);

        auto [logPolicy, valuePred] = model->forward(states, captures);
        auto pLoss = -(policies * logPolicy).sum(1).mean();
        auto vLoss = torch::mse_loss(valuePred, values);
        totalLoss += (pLoss + VALUE_LOSS_WEIGHT * vLoss).item<double>();
        numBatches++;
    }
    return totalLoss / numBatches;
}

static void trainModel(AlphaNet &model, const ReplayBuffer &buf, int gradientSteps,
                       torch::Device device) {
    model->train();
    model->to(device);

    // Move entire buffer to device once up front
    auto states   = buf.states.to(device);
    auto captures = buf.captures.to(device);
    auto policies = buf.policies.to(device);
    auto values   = buf.values.to(device);

    torch::optim::SGD optimizer(
        model->parameters(),
        torch::optim::SGDOptions(LR).momentum(0.9).weight_decay(WEIGHT_DECAY));

    int64_t n             = buf.size();
    double  totalPolicy   = 0.0;
    double  totalValue    = 0.0;
    int     reportEvery   = std::max(1, gradientSteps / 10);

    std::cout << "  baseline policy loss (uniform): ~5.89\n";

    for (int step = 0; step < gradientSteps; step++) {
        auto idx = torch::randint(0, n, {BATCH_SIZE}, torch::kInt64).to(device);

        auto bStones   = states.index_select(0, idx);
        auto bCaptures = captures.index_select(0, idx);
        auto bPolicies = policies.index_select(0, idx);
        auto bValues   = values.index_select(0, idx);
        augmentBatch(bStones, bCaptures, bPolicies, bValues);
        auto bStates = decodeStates(bStones, bCaptures);
        bPolicies    = bPolicies.to(torch::kFloat);

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
    bool        bootstrap     = false;
    float       valueAlpha    = 0.2f;
    std::string outDirFlag    = "";  // "" = checkpoints/<game>

    auto usage = [&](std::ostream &out) {
        out <<
            "Usage: train [-g game] [-t steps] [-a alpha] [-d dir] [-b]\n"
            "\n"
            "Options:\n"
            "  -g  game: pente | gomoku | keryopente      (default: " << gameFlag << ")\n"
            "  -t  gradient steps (0 = auto from buffer)  (default: " << stepsOverride << ")\n"
            "  -a  value blend alpha: target = a*z + (1-a)*rootQ  (default: " << valueAlpha << ")\n"
            "  -d  checkpoint output dir                  (default: checkpoints/<game>;\n"
            "      the buffer is always read from checkpoints/<game> — lets alpha-sweep\n"
            "      runs share one buffer without clobbering each other's best_model.pt)\n"
            "  -b  bootstrap mode — train from bootstrap.pt instead of buffer.pt\n"
            "\n"
            "Examples:\n"
            "  # as called by train_loop.sh during self-play training\n"
            "  ./train -g pente\n"
            "\n"
            "  # ad hoc: train with a fixed step budget\n"
            "  ./train -g pente -t 500\n"
            "\n"
            "  # alpha-sweep experiment against shared bootstrap data\n"
            "  ./train -g pente -b -a 0.4 -d checkpoints/pente_a40\n";
    };

    int opt;
    while ((opt = getopt(argc, argv, "g:t:a:d:bh")) != -1) {
        if      (opt == 'g') gameFlag      = optarg;
        else if (opt == 't') stepsOverride = std::stoi(optarg);
        else if (opt == 'a') valueAlpha    = std::stof(optarg);
        else if (opt == 'd') outDirFlag    = optarg;
        else if (opt == 'b') bootstrap     = true;
        else if (opt == 'h') { usage(std::cout); return 0; }
        else                 { usage(std::cerr); return 1; }
    }

    const std::string bufDir     = std::string(PROJECT_ROOT) + "/checkpoints/" + gameFlag;
    const std::string ckptDir    = outDirFlag.empty() ? bufDir
                                                      : std::string(PROJECT_ROOT) + "/" + outDirFlag;
    const std::string bestPath   = ckptDir + "/best_model.pt";
    const std::string bufferPath = bootstrap ? bufDir + "/bootstrap.pt"
                                             : bufDir + "/buffer.pt";
    std::filesystem::create_directories(ckptDir);

    auto buf = loadBuffer(bufferPath);

    std::cout << "AlphaPente Train\n"
              << "  game  : " << gameFlag << "\n"
              << "  mode  : " << (bootstrap ? "bootstrap" : "selfplay") << "\n"
              << "  alpha : " << valueAlpha << "\n"
              << "  buffer: " << buf.size() << " positions\n\n";

    if (buf.size() < MIN_BUFFER_SIZE) {
        std::cout << "Buffer (" << buf.size() << ") < min (" << MIN_BUFFER_SIZE
                  << ") — skipping training.\n";
        return 0;
    }

    buf.values = blendValueTargets(buf.values, valueAlpha);

    auto [trainBuf, valBuf] = splitBuffer(buf, VAL_FRACTION);

    int gradientSteps = stepsOverride > 0
        ? stepsOverride
        : std::max(MIN_GRAD_STEPS, (int)(trainBuf.size() / BATCH_SIZE));

    std::cout << "── Training (" << gradientSteps << " steps, " << valBuf.size()
              << " positions held out for validation) ──────────────────────────\n";

    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    std::cout << "  device: " << device << "\n\n";

    AlphaNet model(AlphaNetImpl::kChannels, AlphaNetImpl::kResBlocks);
    bool hasBaseline = std::filesystem::exists(bestPath);
    if (hasBaseline)
        torch::load(model, bestPath);

    double baselineValLoss = hasBaseline
        ? computeValLoss(model, valBuf, device)
        : std::numeric_limits<double>::infinity();

    trainModel(model, trainBuf, gradientSteps, device);

    double candidateValLoss = computeValLoss(model, valBuf, device);
    bool promote = !hasBaseline || candidateValLoss <= baselineValLoss * VAL_GATE_TOLERANCE;

    std::cout << "\n── Validation gate ─────────────────────────────────────────────\n";
    if (hasBaseline)
        std::cout << "  baseline val loss : " << baselineValLoss << "\n"
                  << "  candidate val loss: " << candidateValLoss << "\n";
    else
        std::cout << "  no baseline yet — promoting unconditionally\n";
    std::cout << "  → " << (promote ? "PROMOTED" : "REJECTED — best_model.pt unchanged") << "\n";

    int iterNum = nextIterNumber(ckptDir);
    char iterSuffix[8];
    std::snprintf(iterSuffix, sizeof(iterSuffix), "%04d", iterNum);
    std::string iterPath = ckptDir + "/model_iter" + iterSuffix + ".pt";

    model->to(torch::kCPU);
    torch::save(model, iterPath);
    if (promote) torch::save(model, bestPath);

    std::cout << "\n── Saved ────────────────────────────────────────────────────────\n"
              << "  " << iterPath << (promote ? "" : "  (not promoted)") << "\n";
    if (promote) std::cout << "  " << bestPath << "\n";

    return 0;
}

#else

#include <iostream>
int main() {
    std::cout << "LibTorch not available — build with Torch to enable training.\n";
    return 1;
}

#endif
