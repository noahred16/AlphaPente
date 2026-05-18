#include "NNModel.hpp"
#include <filesystem>
#include <iostream>

int main() {
    std::cout << "AlphaPente — model initialization" << std::endl;

#ifdef WITH_TORCH
    std::string outDir = PROJECT_ROOT "/checkpoints/pente";
    std::filesystem::create_directories(outDir);
    std::string path = outDir + "/best_model.pt";

    AlphaNet model(64, 5);
    torch::save(model, path);

    size_t params = 0;
    for (const auto &p : model->parameters())
        params += static_cast<size_t>(p.numel());

    std::cout << "Saved initialized AlphaNet to " << path << std::endl;
    std::cout << "  Channels: 64, ResBlocks: 5, Parameters: " << params << std::endl;
#else
    std::cout << "LibTorch not available — build with Torch to enable training." << std::endl;
#endif

    return 0;
}
