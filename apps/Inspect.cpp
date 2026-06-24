#ifdef WITH_TORCH
#include "NNModel.hpp"
#include "PenteGame.hpp"
#include "GameUtils.hpp"
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <torch/torch.h>
#include <unistd.h>

static void printExample(int idx,
                          torch::Tensor planes,   // [3, 19, 19]
                          torch::Tensor policy,   // [361]
                          float value) {
    constexpr int B = PenteGame::BOARD_SIZE;
    auto pAcc = planes.accessor<float, 3>();
    auto polAcc = policy.accessor<float, 1>();

    std::cout << "── Example " << idx << "  value: " << std::fixed
              << std::setprecision(1) << value << " ──────────────────\n";

    // Board: plane 0 = current player (○), plane 1 = opponent (●)
    const char *cols = "A B C D E F G H J K L M N O P Q R S T";
    std::cout << "   " << cols << "\n";
    for (int y = B - 1; y >= 0; y--) {
        std::cout << std::setw(2) << (y + 1) << " ";
        for (int x = 0; x < B; x++) {
            if      (pAcc[0][y][x] > 0.5f) std::cout << "○ ";
            else if (pAcc[1][y][x] > 0.5f) std::cout << "● ";
            else                            std::cout << "· ";
        }
        std::cout << (y + 1) << "\n";
    }
    std::cout << "   " << cols << "\n";

    // Top 8 policy moves
    std::vector<std::pair<float, int>> moves;
    for (int i = 0; i < B * B; i++)
        if (polAcc[i] > 0.0f)
            moves.emplace_back(polAcc[i], i);
    std::sort(moves.rbegin(), moves.rend());

    std::cout << "  policy (" << moves.size() << " moves visited):";
    for (int i = 0; i < std::min((int)moves.size(), 8); i++) {
        int cell = moves[i].second;
        std::cout << "  " << GameUtils::displayMove(cell % B, cell / B)
                  << "(" << std::setprecision(2) << moves[i].first << ")";
    }
    std::cout << "\n\n";
}

// Runs the model on up to sampleSize evenly-spaced buffer positions and prints
// prediction distribution, MAE, and Pearson correlation vs. training labels.
static void probeValueHead(const std::string &modelPath,
                            torch::Tensor states,
                            torch::Tensor captures,
                            torch::Tensor labels,
                            int64_t sampleSize) {
    AlphaNet model(AlphaNetImpl::kChannels, AlphaNetImpl::kResBlocks);
    torch::load(model, modelPath);
    model->eval();

    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    model->to(device);

    int64_t N = states.size(0);
    if (sampleSize > 0 && sampleSize < N) {
        auto idx = torch::linspace(0, (float)(N - 1), sampleSize).to(torch::kInt64);
        states   = states.index_select(0, idx);
        captures = captures.index_select(0, idx);
        labels   = labels.index_select(0, idx);
        N = sampleSize;
    }

    constexpr int64_t CHUNK = 512;
    std::vector<float> preds, labs;
    preds.reserve(N);
    labs.reserve(N);

    torch::NoGradGuard no_grad;
    for (int64_t i = 0; i < N; i += CHUNK) {
        int64_t end = std::min(i + CHUNK, N);
        auto [_, v] = model->forward(
            states.slice(0, i, end).to(device),
            captures.slice(0, i, end).to(device));
        auto vFlat = v.squeeze(1).to(torch::kFloat).cpu();
        auto lFlat = labels.slice(0, i, end).squeeze(1).contiguous();
        auto vAcc = vFlat.accessor<float, 1>();
        auto lAcc = lFlat.accessor<float, 1>();
        for (int64_t j = 0; j < end - i; j++) {
            preds.push_back(vAcc[j]);
            labs.push_back(lAcc[j]);
        }
    }

    double sumP = 0, sumL = 0, sumAE = 0;
    for (int64_t i = 0; i < N; i++) { sumP += preds[i]; sumL += labs[i]; sumAE += std::abs(preds[i] - labs[i]); }
    double meanP = sumP / N, meanL = sumL / N, mae = sumAE / N;

    double varP = 0, varL = 0, cov = 0;
    for (int64_t i = 0; i < N; i++) {
        double dp = preds[i] - meanP, dl = labs[i] - meanL;
        varP += dp * dp; varL += dl * dl; cov += dp * dl;
    }
    double stdP = std::sqrt(varP / N), stdL = std::sqrt(varL / N);
    double pearsonR = (stdP > 1e-6 && stdL > 1e-6) ? cov / N / (stdP * stdL) : 0.0;

    int bins[5] = {};
    for (float p : preds) {
        if      (p < -0.6f) bins[0]++;
        else if (p < -0.2f) bins[1]++;
        else if (p <  0.2f) bins[2]++;
        else if (p <  0.6f) bins[3]++;
        else                bins[4]++;
    }

    std::cout << "\n── Value Head Probe (" << N << " positions from " << modelPath << ") ──\n"
              << std::fixed << std::setprecision(3)
              << "  mean pred:  " << std::setw(7) << meanP << "   labels: " << meanL << "\n"
              << "  std  pred:  " << std::setw(7) << stdP  << "   labels: " << stdL  << "\n"
              << "  MAE:        " << std::setw(7) << mae   << "\n"
              << "  Pearson r:  " << std::setw(7) << pearsonR << "\n"
              << "\n  Prediction distribution:\n"
              << "    [-1.0,-0.6): " << bins[0] << "  (" << std::setprecision(1) << 100.0*bins[0]/N << "%)\n"
              << "    [-0.6,-0.2): " << bins[1] << "  (" << 100.0*bins[1]/N << "%)\n"
              << "    [-0.2, 0.2): " << bins[2] << "  (" << 100.0*bins[2]/N << "%)\n"
              << "    [ 0.2, 0.6): " << bins[3] << "  (" << 100.0*bins[3]/N << "%)\n"
              << "    [ 0.6, 1.0]: " << bins[4] << "  (" << 100.0*bins[4]/N << "%)\n";
}

int main(int argc, char *argv[]) {
    std::string bufPath   = PROJECT_ROOT "/checkpoints/pente/buffer.pt";
    std::string modelPath = "";
    int         n         = 5;
    int64_t     probeSz   = 2000;

    int opt;
    while ((opt = getopt(argc, argv, "b:n:m:s:")) != -1) {
        if      (opt == 'b') bufPath   = optarg;
        else if (opt == 'n') n         = std::stoi(optarg);
        else if (opt == 'm') modelPath = optarg;
        else if (opt == 's') probeSz   = std::stoi(optarg);
    }

    if (!std::filesystem::exists(bufPath)) {
        std::cerr << "Buffer not found: " << bufPath << "\n";
        return 1;
    }

    torch::serialize::InputArchive ar;
    ar.load_from(bufPath);
    torch::Tensor states, captures, policies, values;
    ar.read("states",   states);
    ar.read("captures", captures);
    ar.read("policies", policies);
    ar.read("values",   values);

    int64_t total = states.size(0);
    std::cout << "Buffer: " << total << " positions\n";
    std::cout << "Value distribution:  +1=" << (values > 0.5f).sum().item<int64_t>()
              << "  -1=" << (values < -0.5f).sum().item<int64_t>()
              << "  0="  << (values.abs() < 0.5f).sum().item<int64_t>() << "\n\n";

    // Sample n evenly-spaced examples
    for (int i = 0; i < n; i++) {
        int64_t idx = (total - 1) * i / std::max(n - 1, 1);
        printExample((int)idx,
                     states[idx],
                     policies[idx],
                     values[idx].item<float>());
    }

    if (!modelPath.empty()) {
        if (!std::filesystem::exists(modelPath)) {
            std::cerr << "Model not found: " << modelPath << "\n";
            return 1;
        }
        probeValueHead(modelPath, states, captures, values, probeSz);
    }

    return 0;
}

#else
#include <iostream>
int main() { std::cout << "LibTorch not available.\n"; return 1; }
#endif
