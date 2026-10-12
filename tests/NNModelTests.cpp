#include "doctest.h"

#ifdef WITH_TORCH

#include "NNModel.hpp"
#include <filesystem>
#include <torch/torch.h>

TEST_CASE("AlphaNetImpl - custom channels/resBlocks produces correctly shaped outputs") {
    constexpr int B = AlphaNetImpl::BOARD;
    AlphaNet model(32, 2);  // deliberately non-default size
    model->eval();

    auto planes   = torch::zeros({3, AlphaNetImpl::kInputPlanes, B, B});
    auto captures = torch::zeros({3, 2});

    torch::NoGradGuard noGrad;
    auto [logPolicy, value] = model->forward(planes, captures);

    CHECK(logPolicy.size(0) == 3);
    CHECK(logPolicy.size(1) == B * B);
    CHECK(value.size(0) == 3);
    CHECK(value.size(1) == 1);
}

TEST_CASE("AlphaNetImpl - save/load round trip preserves a non-default architecture") {
    constexpr int B = AlphaNetImpl::BOARD;
    AlphaNet saved(32, 2);
    saved->eval();

    auto planes   = torch::rand({1, AlphaNetImpl::kInputPlanes, B, B});
    auto captures = torch::rand({1, 2});

    torch::NoGradGuard noGrad;
    auto [expectedPolicy, expectedValue] = saved->forward(planes, captures);

    auto path = std::filesystem::temp_directory_path() / "alphanet_test_roundtrip.pt";
    torch::save(saved, path.string());

    AlphaNet loaded(32, 2);  // must match the saved shape
    torch::load(loaded, path.string());
    loaded->eval();
    std::filesystem::remove(path);

    auto [loadedPolicy, loadedValue] = loaded->forward(planes, captures);

    CHECK(torch::allclose(expectedPolicy, loadedPolicy));
    CHECK(torch::allclose(expectedValue, loadedValue));
}

// torch::load does NOT error on a channels mismatch — it silently resizes the
// destination tensors to the archive's shape instead of keeping the requested
// one. checkArchitectureMatches exists specifically to catch that; this test
// pins the underlying torch::load behavior so if a future libtorch version
// changes it (e.g. starts throwing on its own), that's visible here too.
TEST_CASE("AlphaNetImpl - torch::load silently resizes on a channels mismatch") {
    AlphaNet saved(32, 2);
    saved->eval();

    auto path = std::filesystem::temp_directory_path() / "alphanet_test_channels_mismatch.pt";
    torch::save(saved, path.string());

    AlphaNet mismatched(64, 2);  // different channel count than what was saved
    torch::load(mismatched, path.string());

    CHECK(mismatched->inputConv->weight.size(0) == 32);  // silently became 32, not the requested 64
    CHECK_THROWS_AS(checkArchitectureMatches(mismatched, 64, path.string()), std::runtime_error);

    std::filesystem::remove(path);
}

// A res_blocks (topology) mismatch is different: it changes which named
// submodules exist, so torch::load itself throws without any extra check.
TEST_CASE("AlphaNetImpl - torch::load throws on its own for a resBlocks count mismatch") {
    AlphaNet saved(32, 2);
    saved->eval();

    auto path = std::filesystem::temp_directory_path() / "alphanet_test_resblocks_mismatch.pt";
    torch::save(saved, path.string());

    AlphaNet mismatched(32, 4);  // different res block count than what was saved
    CHECK_THROWS(torch::load(mismatched, path.string()));

    std::filesystem::remove(path);
}

#endif // WITH_TORCH
