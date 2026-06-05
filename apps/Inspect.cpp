#ifdef WITH_TORCH
#include "PenteGame.hpp"
#include "GameUtils.hpp"
#include <filesystem>
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

int main(int argc, char *argv[]) {
    std::string bufPath = PROJECT_ROOT "/checkpoints/pente/buffer.pt";
    int n = 5;

    int opt;
    while ((opt = getopt(argc, argv, "b:n:")) != -1) {
        if      (opt == 'b') bufPath = optarg;
        else if (opt == 'n') n = std::stoi(optarg);
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

    return 0;
}

#else
#include <iostream>
int main() { std::cout << "LibTorch not available.\n"; return 1; }
#endif
