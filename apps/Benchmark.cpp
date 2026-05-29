#include "Evaluator.hpp"
#include "GameUtils.hpp"
#include "PenteGame.hpp"
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>

// ── JSON loader ───────────────────────────────────────────────────────────────
// Minimal parser for the suite format: [{"state": "...", "expected": ["A1","B2"]}, ...]

struct TestCase {
    std::string state;
    std::vector<std::string> expected;
};

static std::vector<TestCase> loadSuite(const std::string &path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "Cannot open suite: " << path << "\n";
        return {};
    }

    std::vector<TestCase> cases;
    TestCase cur;
    bool inExpected = false;
    std::string line;

    while (std::getline(f, line)) {
        if (line.find("\"state\"") != std::string::npos) {
            size_t colon = line.find(':');
            size_t q1    = line.find('"', colon + 1);
            size_t q2    = (q1 != std::string::npos) ? line.find('"', q1 + 1) : std::string::npos;
            if (q2 != std::string::npos)
                cur.state = line.substr(q1 + 1, q2 - q1 - 1);
            inExpected = false;
        } else if (line.find("\"expected\"") != std::string::npos) {
            cur.expected.clear();
            inExpected = true;
        } else if (inExpected) {
            size_t q1 = line.find('"');
            size_t q2 = (q1 != std::string::npos) ? line.find('"', q1 + 1) : std::string::npos;
            if (q2 != std::string::npos && q2 > q1 + 1)
                cur.expected.push_back(line.substr(q1 + 1, q2 - q1 - 1));
            if (line.find(']') != std::string::npos) {
                inExpected = false;
                cases.push_back(cur);
                cur = {};
            }
        }
    }

    return cases;
}

// ── CSV (append one row per run) ──────────────────────────────────────────────

static std::string isoTimestamp() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", std::localtime(&t));
    return buf;
}

static void appendResult(const std::string &csvPath,
                         const std::string &checkpoint,
                         const std::string &evaluatorName,
                         const std::string &suiteName,
                         int passed, int total) {
    std::filesystem::create_directories(std::filesystem::path(csvPath).parent_path());
    bool needsHeader = !std::filesystem::exists(csvPath);
    std::ofstream f(csvPath, std::ios::app);
    if (needsHeader)
        f << "timestamp,checkpoint,evaluator,suite,passed,total,score_pct\n";
    double pct = total > 0 ? 100.0 * passed / total : 0.0;
    f << isoTimestamp() << ","
      << checkpoint << ","
      << evaluatorName << ","
      << suiteName << ","
      << passed << ","
      << total << ","
      << std::fixed << std::setprecision(1) << pct << "\n";
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char *argv[]) {
    std::string modelPath = PROJECT_ROOT "/checkpoints/pente/best_model.pt";
    std::string suitePath = PROJECT_ROOT "/tests/open-three-suite.json";
    std::string outPath   = PROJECT_ROOT "/reports/pente/benchmark.csv";

    int opt;
    while ((opt = getopt(argc, argv, "p:t:o:")) != -1) {
        if      (opt == 'p') modelPath = optarg;
        else if (opt == 't') suitePath = optarg;
        else if (opt == 'o') outPath   = optarg;
    }

    std::cout << "AlphaPente Benchmark\n"
              << "  model : " << modelPath << "\n"
              << "  suite : " << suitePath << "\n"
              << "  output: " << outPath << "\n\n";

    auto cases = loadSuite(suitePath);
    if (cases.empty()) {
        std::cerr << "No test cases loaded.\n";
        return 1;
    }
    std::cout << "Loaded " << cases.size() << " test cases\n\n";

    HeuristicEvaluator heuristicEval;
    Evaluator *evaluator = &heuristicEval;
    std::string evaluatorName = "heuristic";

#ifdef WITH_TORCH
    std::unique_ptr<NNEvaluator> nnEval;
    if (modelPath != "heuristic") {
        if (std::filesystem::exists(modelPath)) {
            nnEval = std::make_unique<NNEvaluator>(modelPath);
            evaluator = nnEval.get();
            evaluatorName = "nn";
            std::cout << "Using NNEvaluator\n\n";
        } else {
            std::cout << "Model not found — falling back to HeuristicEvaluator\n\n";
        }
    } else {
        std::cout << "Using HeuristicEvaluator\n\n";
    }
#else
    std::cout << "LibTorch not available — using HeuristicEvaluator\n\n";
#endif

    int passed = 0;
    int total  = static_cast<int>(cases.size());

    for (int i = 0; i < total; ++i) {
        const auto &tc = cases[i];

        PenteGame game(PenteGame::Config::pente());
        game.reset();
        for (const auto &mv : GameUtils::parseGameString(tc.state.c_str()))
            game.makeMove(mv.c_str());

        auto policy = evaluator->evaluatePolicy(game);

        std::string topMove;
        if (!policy.empty())
            topMove = GameUtils::displayMove(policy.front().first.x, policy.front().first.y);

        for (const auto &exp : tc.expected)
            if (topMove == exp) { ++passed; break; }

        if ((i + 1) % 50 == 0 || i + 1 == total)
            std::cout << "  " << (i + 1) << "/" << total
                      << "  running: " << passed << "/" << (i + 1) << "\n";
    }

    double pct = 100.0 * passed / total;
    std::cout << "\nResult: " << passed << "/" << total
              << "  (" << std::fixed << std::setprecision(1) << pct << "%)\n";

    std::string suiteName = std::filesystem::path(suitePath).stem().string();
    appendResult(outPath, modelPath, evaluatorName, suiteName, passed, total);
    std::cout << "Appended to " << outPath << "\n";

    return 0;
}
