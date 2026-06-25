#include "Evaluator.hpp"
#include "GameUtils.hpp"
#include "ParallelMCTS.hpp"
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

// ── Arena (NN vs Heuristic) ───────────────────────────────────────────────────

struct ArenaResult { int nnWins, hWins, draws; };

static ArenaResult runArena(Evaluator *candidateEval, Evaluator *opponentEval,
                             bool opponentIsNN, const std::string &opponentLabel,
                             int numGames, int candidateSims, int opponentSims,
                             const PenteGame::Config &gameConfig) {
    ArenaResult result{0, 0, 0};

    ParallelMCTS::Config candCfg, oppCfg;
    candCfg.maxIterations       = candidateSims;
    candCfg.explorationConstant = 1.7;
    candCfg.numWorkerThreads    = 6;
    candCfg.numEvalThreads      = 1;
    candCfg.arenaSize           = GameUtils::arenaSizeFromEnv();
    candCfg.evaluator           = candidateEval;

    oppCfg.maxIterations        = opponentSims;
    oppCfg.explorationConstant  = 1.7;
    oppCfg.numWorkerThreads     = 6;
    oppCfg.numEvalThreads       = opponentIsNN ? 1 : 0;
    oppCfg.arenaSize            = GameUtils::arenaSizeFromEnv();
    oppCfg.evaluator            = opponentEval;

    for (int g = 0; g < numGames; g++) {
        bool candIsBlack = (g % 2 == 0);

        ParallelMCTS candMcts(candCfg), oppMcts(oppCfg);
        PenteGame game(gameConfig);
        game.reset();

        while (!game.isGameOver()) {
            bool isBlackTurn = (game.getCurrentPlayer() == PenteGame::BLACK);
            bool candTurn    = (candIsBlack == isBlackTurn);
            auto &mcts       = candTurn ? candMcts : oppMcts;

            mcts.search(game);
            PenteGame::Move move = mcts.getBestMove();
            game.makeMove(move.x, move.y);
            candMcts.reuseSubtree(move);
            oppMcts.reuseSubtree(move);
        }

        PenteGame::Player winner = game.getWinner();
        bool candWon = (candIsBlack  && winner == PenteGame::BLACK) ||
                       (!candIsBlack && winner == PenteGame::WHITE);
        bool draw    = (winner == PenteGame::NONE);

        if      (draw)     result.draws++;
        else if (candWon)  result.nnWins++;
        else               result.hWins++;

        std::string candColor = candIsBlack ? "B" : "W";
        std::string outcome   = draw ? "draw" : (candWon ? "nn" : opponentLabel);
        std::cout << "  game " << std::setw(2) << (g + 1)
                  << "  nn=" << candColor
                  << "  winner: " << outcome
                  << "  (nn " << result.nnWins << "/"
                  << (g + 1 - result.draws) << " decisive)\n";
    }

    return result;
}

// ── Suite runner ──────────────────────────────────────────────────────────────

struct SuiteResult { int passed, total; float sumAbsValue; };

static SuiteResult runSuiteCheck(Evaluator *evaluator, ParallelMCTS *suiteMcts,
                                  const std::string &suitePath, bool valueMode, bool verbose) {
    auto cases = loadSuite(suitePath);
    if (cases.empty()) return {0, 0, 0.0f};
    std::cout << "Loaded " << cases.size() << " test cases\n\n";

    int   passed      = 0;
    int   total       = static_cast<int>(cases.size());
    float sumAbsValue = 0.0f;

    for (int i = 0; i < total; ++i) {
        const auto &tc = cases[i];

        PenteGame game(PenteGame::Config::pente());
        game.reset();
        for (const auto &mv : GameUtils::parseGameString(tc.state.c_str()))
            game.makeMove(mv.c_str());

        if (valueMode) {
            float value = evaluator->evaluateValue(game);
            sumAbsValue += std::abs(value);
            // Convention: value is from previous-player perspective (+1 = mover wins).
            // "win" = current player wins = previous player (mover) LOST → value < 0.
            bool expectWin = (!tc.expected.empty() && tc.expected[0] == "win");
            bool correct   = expectWin ? (value < 0.0f) : (value > 0.0f);
            if (correct) ++passed;
        } else {
            std::string topMove;
#ifdef WITH_TORCH
            if (suiteMcts) {
                suiteMcts->reset();
                suiteMcts->search(game);
                PenteGame::Move mv = suiteMcts->getBestMove();
                topMove = GameUtils::displayMove(mv.x, mv.y);
            } else
#endif
            {
                auto policy = evaluator->evaluatePolicy(game);
                if (!policy.empty())
                    topMove = GameUtils::displayMove(policy.front().first.x, policy.front().first.y);
            }
            bool ok = false;
            for (const auto &exp : tc.expected)
                if (topMove == exp) { ok = true; break; }
            if (ok) {
                ++passed;
            } else if (verbose) {
                std::cout << "  FAIL [" << (i + 1) << "] state=" << tc.state
                          << "  expected=";
                for (const auto &e : tc.expected) std::cout << e << " ";
                std::cout << " got=" << topMove << "\n";
            }
        }

        if (!verbose && ((i + 1) % 50 == 0 || i + 1 == total))
            std::cout << "  " << (i + 1) << "/" << total
                      << "  running: " << passed << "/" << (i + 1) << "\n";
    }

    return {passed, total, sumAbsValue};
}

// ── main ──────────────────────────────────────────────────────────────────────

static std::string latestCheckpoint(const std::string &ckptDir) {
    std::string best;
    int maxIter = 0;
    if (std::filesystem::exists(ckptDir)) {
        for (const auto &entry : std::filesystem::directory_iterator(ckptDir)) {
            const std::string stem = entry.path().stem().string();
            if (stem.rfind("model_iter", 0) == 0) {
                try {
                    int n = std::stoi(stem.substr(10));
                    if (n > maxIter) { maxIter = n; best = entry.path().string(); }
                } catch (...) {}
            }
        }
    }
    if (!best.empty()) return best;
    return ckptDir + "/best_model.pt";  // fallback before first training run
}

int main(int argc, char *argv[]) {
    std::string gameFlag  = "pente";
    bool runValueSuite    = false;
    bool verbose          = false;

    // Pre-scan for -g, -V, -v so default paths are correct
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "-g" && i + 1 < argc) gameFlag = argv[i + 1];
        if (std::string(argv[i]) == "-V") runValueSuite = true;
        if (std::string(argv[i]) == "-v") verbose = true;
    }

    const std::string ckptDir        = std::string(PROJECT_ROOT) + "/checkpoints/" + gameFlag;
    std::string       modelPath      = latestCheckpoint(ckptDir);
    std::string       outPath        = std::string(PROJECT_ROOT) + "/reports/" + gameFlag + "/benchmark.csv";
    const std::string openThreePath  = std::string(PROJECT_ROOT) + "/tests/open-three-suite.json";
    const std::string valueSuitePath = std::string(PROJECT_ROOT) + "/tests/value-suite.json";
    std::string       suitePath      = runValueSuite ? valueSuitePath : openThreePath;

    bool        runArenaFlag = false;
    bool        fullMode     = true;  // disabled when any mode flag is given explicitly
    int         arenaGames   = 10;
    int         arenaSims    = 800;
    int         opponentSims = 0;     // 0 = same as arenaSims
    int         suiteSims    = 0;     // 0 = raw policy, >0 = MCTS with N sims
    std::string opponentPath;

    auto resolve = [](const std::string &path) {
        if (std::filesystem::exists(path)) return path;
        std::string rooted = std::string(PROJECT_ROOT) + "/" + path;
        return std::filesystem::exists(rooted) ? rooted : path;
    };

    auto usage = [&](std::ostream &out) {
        out <<
            "Usage: benchmark [-g game] [-p model] [-t suite] [-o out] [-s sims] [-V] [-a] [-G games] [-S sims] [-T opp_sims] [-P opponent]\n"
            "\n"
            "Options:\n"
            "  -g  game: pente | gomoku | keryopente      (default: " << gameFlag << ")\n"
            "  -p  candidate model checkpoint              (default: latest in checkpoints/<game>/)\n"
            "  -t  test suite JSON path                    (default: tests/open-three-suite.json)\n"
            "  -o  CSV output path                         (default: reports/<game>/benchmark.csv)\n"
            "  -s  suite MCTS sims per position (0=raw policy) (default: " << suiteSims << ")\n"
            "  -V  value suite: check value sign instead of top move (default suite: tests/value-suite.json)\n"
            "  -a  run arena match after the suite\n"
            "  -G  arena game count                        (default: " << arenaGames << ")\n"
            "  -S  candidate (NN) sims per move            (default: " << arenaSims << ")\n"
            "  -T  opponent sims per move                  (default: same as -S)\n"
            "  -P  opponent model path                     (default: heuristic evaluator)\n"
            "\n"
            "Default (no mode flags): full battery — policy suite (raw + MCTS@800), value\n"
            "  suite, and arena vs heuristic@800/10000/100000.\n"
            "\n"
            "Examples:\n"
            "  # full default battery\n"
            "  ./benchmark -g pente\n"
            "\n"
            "  # suite with MCTS search instead of raw policy\n"
            "  ./benchmark -g pente -s 800\n"
            "\n"
            "  # check value head sign accuracy\n"
            "  ./benchmark -g pente -V\n"
            "\n"
            "  # arena: NN vs heuristic\n"
            "  ./benchmark -g pente -a -G 10 -S 400 -T 100\n"
            "\n"
            "  # ad hoc: pit the latest checkpoint against a roster model\n"
            "  ./benchmark -g pente -a -G 20 -S 400 -P checkpoints/pente/roster/model_iter0030.pt\n";
    };

    int opt;
    while ((opt = getopt(argc, argv, "g:p:t:o:s:VvaG:S:T:P:h")) != -1) {
        if      (opt == 'g') { /* already handled above */ }
        else if (opt == 'p') modelPath    = resolve(optarg);
        else if (opt == 't') { suitePath  = resolve(optarg); fullMode = false; }
        else if (opt == 'o') outPath      = resolve(optarg);
        else if (opt == 's') { suiteSims  = std::stoi(optarg); fullMode = false; }
        else if (opt == 'V') fullMode     = false;
        else if (opt == 'v') { /* already handled above */ }
        else if (opt == 'a') { runArenaFlag = true; fullMode = false; }
        else if (opt == 'G') arenaGames   = std::stoi(optarg);
        else if (opt == 'S') arenaSims    = std::stoi(optarg);
        else if (opt == 'T') opponentSims = std::stoi(optarg);
        else if (opt == 'P') opponentPath = resolve(optarg);
        else if (opt == 'h') { usage(std::cout); return 0; }
        else                 { usage(std::cerr); return 1; }
    }

    std::cout << "AlphaPente Benchmark\n"
              << "  model : " << modelPath << "\n"
              << "  output: " << outPath << "\n\n";

    // Compute relative model path for CSV once
    std::string relModelPath = modelPath;
    const std::string projectRoot = PROJECT_ROOT;
    if (relModelPath.rfind(projectRoot, 0) == 0)
        relModelPath = relModelPath.substr(projectRoot.size() + 1);

    // Load evaluator
    HeuristicEvaluator heuristicEval;
    Evaluator         *evaluator     = &heuristicEval;
    std::string        evaluatorName = "heuristic";

#ifdef WITH_TORCH
    std::unique_ptr<NNEvaluator> nnEval;
    if (modelPath != "heuristic") {
        if (std::filesystem::exists(modelPath)) {
            nnEval        = std::make_unique<NNEvaluator>(modelPath);
            evaluator     = nnEval.get();
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

    // ── Full benchmark battery (default) ─────────────────────────────────────
    if (fullMode) {
        const PenteGame::Config gameConfig =
            (gameFlag == "gomoku")     ? PenteGame::Config::gomoku()     :
            (gameFlag == "keryopente") ? PenteGame::Config::keryoPente() :
                                          PenteGame::Config::pente();

        auto printSuiteResult = [&](const SuiteResult &res, bool valueMode) {
            double pct = 100.0 * res.passed / res.total;
            std::cout << "Result: " << res.passed << "/" << res.total
                      << "  (" << std::fixed << std::setprecision(1) << pct << "%)\n";
            if (valueMode)
                std::cout << "Mean |value|: " << std::fixed << std::setprecision(3)
                          << (res.sumAbsValue / res.total) << "\n";
        };

        // 1. Policy suite — raw policy
        std::cout << "── Policy suite: raw ─────────────────────────────────────────────\n";
        {
            auto res = runSuiteCheck(evaluator, nullptr, openThreePath, false, verbose);
            if (res.total == 0) return 1;
            printSuiteResult(res, false);
            appendResult(outPath, relModelPath, evaluatorName, "open-three-suite", res.passed, res.total);
            std::cout << "Appended to " << outPath << "\n";
        }

#ifdef WITH_TORCH
        if (nnEval) {
            // 2. Policy suite — MCTS 800 sims
            std::cout << "\n── Policy suite: MCTS@800 ────────────────────────────────────────\n";
            {
                ParallelMCTS::Config cfg;
                cfg.maxIterations       = 800;
                cfg.explorationConstant = 1.7;
                cfg.numWorkerThreads    = 6;
                cfg.numEvalThreads      = 1;
                cfg.arenaSize           = GameUtils::arenaSizeFromEnv();
                cfg.evaluator           = nnEval.get();
                cfg.dirichletAlpha      = 0.0f;
                ParallelMCTS mcts800(cfg);
                auto res = runSuiteCheck(nnEval.get(), &mcts800, openThreePath, false, verbose);
                if (res.total == 0) return 1;
                printSuiteResult(res, false);
                appendResult(outPath, relModelPath, "nn@800", "open-three-suite", res.passed, res.total);
                std::cout << "Appended to " << outPath << "\n";
            }

            // 3. Value suite
            std::cout << "\n── Value suite ───────────────────────────────────────────────────\n";
            {
                auto res = runSuiteCheck(nnEval.get(), nullptr, valueSuitePath, true, verbose);
                if (res.total == 0) return 1;
                printSuiteResult(res, true);
                appendResult(outPath, relModelPath, "nn", "value-suite-sign", res.passed, res.total);
                std::cout << "Appended to " << outPath << "\n";
            }

            // 4-6. Arena: NN@800 vs heuristic at increasing sim counts
            HeuristicEvaluator heuristicArenaEval;
            const struct { int sims; const char *label; } arenaOpponents[] = {
                {    800, "heuristic@800"    },
                {  10000, "heuristic@10000"  },
                { 100000, "heuristic@100000" },
            };

            for (const auto &opp : arenaOpponents) {
                std::cout << "\n── Arena: nn@800 vs " << opp.label
                          << " (" << arenaGames << " games) ────────────────────────\n";
                auto ar = runArena(nnEval.get(), &heuristicArenaEval, false, opp.label,
                                   arenaGames, 800, opp.sims, gameConfig);
                int    decisive = ar.nnWins + ar.hWins;
                double nnPct    = decisive > 0 ? 100.0 * ar.nnWins / decisive : 0.0;
                std::cout << "\nArena result: nn " << ar.nnWins
                          << "  " << opp.label << " " << ar.hWins
                          << "  draws " << ar.draws
                          << "  (" << std::fixed << std::setprecision(1) << nnPct << "% nn win rate)\n";
                std::string arenaEval = "nn@800-vs-" + std::string(opp.label);
                appendResult(outPath, relModelPath, arenaEval, "arena", ar.nnWins, arenaGames);
                std::cout << "Appended to " << outPath << "\n";
            }
        }
#endif
        return 0;
    }

    // ── Single-mode run (explicit flags) ─────────────────────────────────────

    std::cout << "  suite : " << suitePath << "\n\n";

    ParallelMCTS *suiteMctsPtr = nullptr;
#ifdef WITH_TORCH
    std::unique_ptr<ParallelMCTS> suiteMcts;
    if (suiteSims > 0 && nnEval) {
        ParallelMCTS::Config cfg;
        cfg.maxIterations       = suiteSims;
        cfg.explorationConstant = 1.7;
        cfg.numWorkerThreads    = 6;
        cfg.numEvalThreads      = 1;
        cfg.arenaSize           = GameUtils::arenaSizeFromEnv();
        cfg.evaluator           = nnEval.get();
        cfg.dirichletAlpha      = 0.0f;  // no noise for benchmarking
        suiteMcts    = std::make_unique<ParallelMCTS>(cfg);
        suiteMctsPtr = suiteMcts.get();
        evaluatorName = "nn@" + std::to_string(suiteSims);
    }
#endif

    auto res = runSuiteCheck(evaluator, suiteMctsPtr, suitePath, runValueSuite, verbose);
    if (res.total == 0) return 1;

    double pct = 100.0 * res.passed / res.total;
    std::cout << "\nResult: " << res.passed << "/" << res.total
              << "  (" << std::fixed << std::setprecision(1) << pct << "%)\n";
    if (runValueSuite)
        std::cout << "Mean |value|: " << std::fixed << std::setprecision(3)
                  << (res.sumAbsValue / res.total) << "\n";

    std::string suiteName = std::filesystem::path(suitePath).stem().string();
    if (runValueSuite) suiteName += "-value-sign";
    appendResult(outPath, relModelPath, evaluatorName, suiteName, res.passed, res.total);
    std::cout << "Appended to " << outPath << "\n";

#ifdef WITH_TORCH
    if (runArenaFlag && nnEval) {
        PenteGame::Config gameConfig =
            (gameFlag == "gomoku")     ? PenteGame::Config::gomoku()     :
            (gameFlag == "keryopente") ? PenteGame::Config::keryoPente() :
                                          PenteGame::Config::pente();

        HeuristicEvaluator heuristicArenaEval;
        std::unique_ptr<NNEvaluator> opponentNNEval;
        Evaluator   *opponentEvalPtr = &heuristicArenaEval;
        bool         opponentIsNN   = false;
        std::string  opponentLabel  = "heuristic";
        bool         arenaReady     = true;

        if (!opponentPath.empty()) {
            if (std::filesystem::exists(opponentPath)) {
                opponentNNEval  = std::make_unique<NNEvaluator>(opponentPath);
                opponentEvalPtr = opponentNNEval.get();
                opponentIsNN    = true;
                opponentLabel   = std::filesystem::path(opponentPath).stem().string();
            } else {
                std::cout << "Opponent not found: " << opponentPath << " — skipping arena.\n";
                arenaReady = false;
            }
        }

        if (arenaReady) {
            int oppSims = opponentSims > 0 ? opponentSims : arenaSims;
            std::cout << "\n── Arena: nn vs " << opponentLabel
                      << " (" << arenaGames << " games, nn=" << arenaSims
                      << " opp=" << oppSims << " sims/move) ──────────────────\n";
            auto ar = runArena(nnEval.get(), opponentEvalPtr, opponentIsNN, opponentLabel,
                               arenaGames, arenaSims, oppSims, gameConfig);
            int    decisive = ar.nnWins + ar.hWins;
            double nnPct    = decisive > 0 ? 100.0 * ar.nnWins / decisive : 0.0;
            std::cout << "\nArena result: nn " << ar.nnWins
                      << "  " << opponentLabel << " " << ar.hWins
                      << "  draws " << ar.draws
                      << "  (" << std::fixed << std::setprecision(1) << nnPct << "% nn win rate)\n";

            std::string arenaLabel = "nn@" + std::to_string(arenaSims) + "-vs-" + opponentLabel + "@" + std::to_string(oppSims);
            appendResult(outPath, relModelPath, arenaLabel, "arena", ar.nnWins, arenaGames);
            std::cout << "Appended arena result to " << outPath << "\n";
        }
    }
#endif

    return 0;
}
