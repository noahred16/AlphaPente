#ifdef WITH_TORCH

#include "GameUtils.hpp"
#include "PenteGame.hpp"
#include "TrainCommon.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <torch/torch.h>
#include <unistd.h>
#include <vector>

// Reconstructs recent self-play games out of a flat position buffer, for
// eyeballing what self-play is actually producing.
//
// Every row is just a board from whoever's-about-to-move's perspective
// (plane 0 = their stones, plane 1 = the opponent's). mergeAndTrim only ever
// appends+trims from the front, never shuffles, and Generate.cpp pushes each
// game's rows in play order before moving to the next game — so the buffer
// is one long chronological tape, and games/moves are recoverable from it:
//   - a game boundary is either a row with zero stones on the board (every
//     game starts empty — always true for buffers with no move_index field)
//     or a discontinuity in move_index (see apps/TrainCommon.hpp's
//     ReplayBuffer) — needed because src/SelfPlay.cpp's exploration-dropoff
//     trimming means a game's first *stored* row is often no longer empty
//   - move_index also gives each game's true starting ply directly, so the
//     mover's color at every row is exact (move 0 is Black), not just
//     "alternates, color unknown" — except for the one leading game that a
//     requested window/ring-buffer-eviction might cut into mid-game, which
//     stays genuinely ambiguous (see `partial` below)
//   - the move between row i and i+1 is whichever cell newly becomes the
//     mover's own stone; any opponent stones that vanish in the same step
//     were captured
constexpr int B = PenteGame::BOARD_SIZE;

struct Cell { int x, y; };
struct PolicyCandidate { Cell cell; float p; };

struct Ply {
    int index;                        // 0 = this game's first visible row
    bool hasMove = false;             // false only for index == 0
    bool moverIsBlack = false;        // color of whoever just moved to reach this ply
    Cell move{};
    std::vector<Cell> captured;
    bool ambiguous = false;           // move reconstruction found != 1 candidate cell
    std::vector<Cell> blackStones, whiteStones;
    float blackCaptures = 0.0f, whiteCaptures = 0.0f;
    float z = 0.0f, rootQ = 0.0f;
    double entropy = 0.0;
    std::vector<PolicyCandidate> topPolicy;
};

struct Game {
    int startPly = 0;      // this game's true move-count at its first stored row (0 if unknown/unavailable)
    bool partial = false;  // leading game whose start the window/ring-buffer cut into — startPly may be wrong
    std::vector<Ply> plies;
};

struct Board {
    bool black[B][B] = {};
    bool white[B][B] = {};
};

static Board absoluteBoard(const torch::TensorAccessor<float, 3> &row, bool blackToMove) {
    Board out;
    for (int y = 0; y < B; y++) {
        for (int x = 0; x < B; x++) {
            bool mine = row[0][y][x] > 0.5f;
            bool opp  = row[1][y][x] > 0.5f;
            if (blackToMove) { out.black[y][x] = mine; out.white[y][x] = opp; }
            else             { out.white[y][x] = mine; out.black[y][x] = opp; }
        }
    }
    return out;
}

// Cells set in `to` but not in `from`.
static std::vector<Cell> added(const bool (&from)[B][B], const bool (&to)[B][B]) {
    std::vector<Cell> cells;
    for (int y = 0; y < B; y++)
        for (int x = 0; x < B; x++)
            if (to[y][x] && !from[y][x]) cells.push_back({x, y});
    return cells;
}

static std::vector<Cell> stoneList(const bool (&grid)[B][B]) {
    std::vector<Cell> cells;
    for (int y = 0; y < B; y++)
        for (int x = 0; x < B; x++)
            if (grid[y][x]) cells.push_back({x, y});
    return cells;
}

static std::vector<PolicyCandidate> topPolicyMoves(const torch::TensorAccessor<float, 1> &pol, int k) {
    std::vector<PolicyCandidate> moves;
    for (int i = 0; i < B * B; i++)
        if (pol[i] > 0.0f) moves.push_back({{i % B, i / B}, pol[i]});
    std::sort(moves.begin(), moves.end(), [](auto &a, auto &b) { return a.p > b.p; });
    if ((int)moves.size() > k) moves.resize(k);
    return moves;
}

static double policyEntropy(const torch::TensorAccessor<float, 1> &pol) {
    double h = 0.0;
    for (int i = 0; i < B * B; i++)
        if (pol[i] > 0.0f) h -= pol[i] * std::log(pol[i]);
    return h;
}

// Walks the sliced buffer tensors and splits them into games. Games are
// returned oldest-first, matching buffer order. `moveIndex` may be an
// undefined tensor (buffer predates that field) — boundary detection and
// move parity fall back to the pure stone-count heuristic in that case.
static std::vector<Game> reconstructGames(torch::Tensor &states, torch::Tensor &captures,
                                           torch::Tensor &policies, torch::Tensor &values,
                                           torch::Tensor &moveIndex,
                                           float capturesToWin, int topK) {
    int64_t n = states.size(0);
    auto stateAcc = states.accessor<float, 4>();
    auto capAcc   = captures.accessor<float, 2>();
    auto polAcc   = policies.accessor<float, 2>();
    auto valAcc   = values.accessor<float, 2>();

    auto stoneCount = states.sum({1, 2, 3});
    auto countAcc = stoneCount.accessor<float, 1>();
    bool firstGamePartial = n > 0 && countAcc[0] > 0.5f;

    torch::Tensor moveIndexContig;
    const float *miPtr = nullptr;
    if (moveIndex.defined()) {
        moveIndexContig = moveIndex.contiguous();
        miPtr = moveIndexContig.data_ptr<float>();
    }

    std::vector<Game> games;

    auto buildGame = [&](int64_t b, int64_t e) {
        Game game;
        game.partial = (b == 0 && firstGamePartial);
        // -1 in moveIndex means "buffer predates this field" (backfilled by
        // mergeAndTrim) — fall back to assuming an ordinary move-0 start.
        if (miPtr && miPtr[b] >= 0.0f) game.startPly = (int)miPtr[b];

        Board prev;
        bool havePrev = false;
        for (int64_t i = b; i < e; i++) {
            int64_t k = i - b;
            int64_t ply = game.startPly + k;
            // Black always opens (ply 0), so true parity comes straight from
            // the absolute ply number once we know where this game starts.
            bool blackToMove = (ply % 2) == 0;
            Board cur = absoluteBoard(stateAcc[i], blackToMove);

            float myCap  = capAcc[i][0] * capturesToWin;
            float oppCap = capAcc[i][1] * capturesToWin;

            Ply plyRec;
            plyRec.index = (int)ply;
            plyRec.blackCaptures = blackToMove ? myCap : oppCap;
            plyRec.whiteCaptures = blackToMove ? oppCap : myCap;
            plyRec.blackStones = stoneList(cur.black);
            plyRec.whiteStones = stoneList(cur.white);
            plyRec.z = valAcc[i][0];
            plyRec.rootQ = valAcc[i][1];
            plyRec.entropy = policyEntropy(polAcc[i]);
            plyRec.topPolicy.clear();
            for (auto &c : topPolicyMoves(polAcc[i], topK))
                plyRec.topPolicy.push_back(c);

            if (havePrev) {
                // Move made by the previous row's mover (opposite color of `cur`'s mover).
                auto moverAdded = blackToMove ? added(prev.white, cur.white)
                                               : added(prev.black, cur.black);
                // Cells that disappeared (set in prev, gone in cur) belong to
                // the non-mover color — those were captured.
                auto captured   = blackToMove ? added(cur.black, prev.black)
                                               : added(cur.white, prev.white);
                plyRec.hasMove = true;
                plyRec.moverIsBlack = !blackToMove;  // previous row's mover
                plyRec.ambiguous = moverAdded.size() != 1;
                plyRec.move = plyRec.ambiguous ? Cell{-1, -1} : moverAdded[0];
                plyRec.captured = captured;
            }

            game.plies.push_back(std::move(plyRec));
            prev = cur;
            havePrev = true;
        }
        games.push_back(std::move(game));
    };

    int64_t gameStart = 0;
    for (int64_t i = 1; i < n; i++) {
        bool stoneBoundary = countAcc[i] < 0.5f;  // empty board => new game's opening row
        // A move_index discontinuity also marks a boundary — needed because
        // a trimmed game's first *stored* row is no longer an empty board.
        // Only trusted when both sides have a real (non-sentinel) value.
        bool moveIdxBoundary = miPtr && miPtr[i] >= 0.0f && miPtr[i - 1] >= 0.0f
                                && miPtr[i] != miPtr[i - 1] + 1.0f;
        if (stoneBoundary || moveIdxBoundary) {
            buildGame(gameStart, i);
            gameStart = i;
        }
    }
    if (n > 0) buildGame(gameStart, n);
    return games;
}

static void printText(const std::vector<Game> &games) {
    int gameNum = 0;
    for (const auto &game : games) {
        gameNum++;
        std::cout << "── Game " << gameNum << (game.partial ? " (partial — cut off by window)" : "")
                   << (game.startPly > 0 ? "  starts at move " + std::to_string(game.startPly) + " (earlier moves trimmed — exploration)" : "")
                   << "  " << game.plies.size() << " row" << (game.plies.size() == 1 ? "" : "s") << " ──\n";
        for (const auto &ply : game.plies) {
            if (ply.hasMove) {
                std::string moveStr = ply.ambiguous ? "?" : GameUtils::displayMove(ply.move.x, ply.move.y);
                std::cout << "  " << std::setw(3) << ply.index << ". " << (ply.moverIsBlack ? "Black " : "White ")
                           << std::left << std::setw(5) << moveStr << std::right;
                if (ply.ambiguous)
                    std::cout << " [ambiguous reconstruction]";
                if (!ply.captured.empty()) {
                    std::cout << " captured:";
                    for (auto &c : ply.captured) std::cout << " " << GameUtils::displayMove(c.x, c.y);
                }
            }
            std::cout << "  z=" << std::fixed << std::setprecision(2) << std::setw(5) << ply.z
                       << " rootQ=" << std::setw(5) << ply.rootQ
                       << " H=" << std::setprecision(3) << ply.entropy
                       << " caps b=" << std::setprecision(0) << ply.blackCaptures << " w=" << ply.whiteCaptures
                       << "  next-to-move top: ";
            for (auto &c : ply.topPolicy)
                std::cout << GameUtils::displayMove(c.cell.x, c.cell.y) << "(" << std::setprecision(2) << c.p << ") ";
            std::cout << "\n";
        }
        std::cout << "\n";
    }
}

static void writeCells(std::ostream &out, const std::vector<Cell> &cells) {
    out << "[";
    for (size_t i = 0; i < cells.size(); i++) {
        if (i) out << ",";
        out << "[" << cells[i].x << "," << cells[i].y << "]";
    }
    out << "]";
}

// games[] is oldest-first (buffer order); JSON numbers them most-recent-first
// (index 0 = most recently played game), matching how the dashboard paginates.
static void printJSON(const std::string &bufPath, int64_t total, const std::vector<Game> &games,
                       float capturesToWin) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(4);
    out << "{\"buffer_path\":\"" << bufPath << "\",\"total_positions\":" << total
        << ",\"captures_to_win\":" << capturesToWin << ",\"games\":[";
    for (size_t gi = 0; gi < games.size(); gi++) {
        const Game &game = games[games.size() - 1 - gi];  // reverse: most recent first
        if (gi) out << ",";
        out << "{\"index\":" << gi << ",\"partial\":" << (game.partial ? "true" : "false")
            << ",\"start_ply\":" << game.startPly << ",\"plies\":[";
        for (size_t pi = 0; pi < game.plies.size(); pi++) {
            const Ply &ply = game.plies[pi];
            if (pi) out << ",";
            out << "{\"ply\":" << ply.index << ",\"move\":";
            if (ply.hasMove) {
                out << "{\"mover\":\"" << (ply.moverIsBlack ? "black" : "white") << "\""
                    << ",\"display\":\"" << (ply.ambiguous ? "?" : GameUtils::displayMove(ply.move.x, ply.move.y)) << "\""
                    << ",\"x\":" << ply.move.x << ",\"y\":" << ply.move.y
                    << ",\"ambiguous\":" << (ply.ambiguous ? "true" : "false")
                    << ",\"captured\":"; writeCells(out, ply.captured);
                out << "}";
            } else {
                out << "null";
            }
            out << ",\"black_stones\":"; writeCells(out, ply.blackStones);
            out << ",\"white_stones\":"; writeCells(out, ply.whiteStones);
            out << ",\"black_captures\":" << ply.blackCaptures
                << ",\"white_captures\":" << ply.whiteCaptures
                << ",\"z\":" << ply.z << ",\"root_q\":" << ply.rootQ
                << ",\"entropy\":" << ply.entropy
                << ",\"top_policy\":[";
            for (size_t ti = 0; ti < ply.topPolicy.size(); ti++) {
                const auto &c = ply.topPolicy[ti];
                if (ti) out << ",";
                out << "{\"display\":\"" << GameUtils::displayMove(c.cell.x, c.cell.y) << "\""
                    << ",\"x\":" << c.cell.x << ",\"y\":" << c.cell.y << ",\"p\":" << c.p << "}";
            }
            out << "]}";
        }
        out << "]}";
    }
    out << "]}\n";
    std::cout << out.str();
}

int main(int argc, char *argv[]) {
    std::string bufPath = PROJECT_ROOT "/checkpoints/pente/buffer.pt";
    int64_t tail = 100;
    float capturesToWin = 10.0f;  // pente default; override with -g for keryopente (15)
    int topK = 3;
    bool jsonOutput = false;

    int opt;
    while ((opt = getopt(argc, argv, "b:n:g:k:j")) != -1) {
        if      (opt == 'b') bufPath       = optarg;
        else if (opt == 'n') tail          = std::stoll(optarg);
        else if (opt == 'g') capturesToWin = std::stof(optarg);
        else if (opt == 'k') topK          = std::stoi(optarg);
        else if (opt == 'j') jsonOutput    = true;
    }

    if (!std::filesystem::exists(bufPath)) {
        std::cerr << "Buffer not found: " << bufPath << "\n";
        return 1;
    }

    auto buf = loadBuffer(bufPath);
    int64_t total = buf.size();
    if (total == 0) {
        std::cerr << "Buffer empty or unloadable: " << bufPath << "\n";
        return 1;
    }

    int64_t start = std::max<int64_t>(0, total - tail);
    auto states   = buf.states.slice(0, start).to(torch::kFloat);   // [n,2,19,19]
    auto captures = buf.captures.slice(0, start).to(torch::kFloat); // [n,2]
    auto policies = buf.policies.slice(0, start).to(torch::kFloat); // [n,361]
    auto values   = buf.values.slice(0, start);                    // [n,2] (z, rootQ)
    torch::Tensor moveIndex;  // left undefined if the buffer predates this field
    if (buf.moveIndex.defined()) moveIndex = buf.moveIndex.slice(0, start).to(torch::kFloat);

    auto games = reconstructGames(states, captures, policies, values, moveIndex, capturesToWin, topK);

    if (jsonOutput) {
        printJSON(bufPath, total, games, capturesToWin);
    } else {
        std::cout << "Buffer: " << bufPath << "\n"
                   << "  total positions: " << total << "  (showing last " << states.size(0) << ")\n\n";
        printText(games);
    }

    return 0;
}

#else
#include <iostream>
int main() {
    std::cout << "LibTorch not available — build with Torch to enable replay.\n";
    return 1;
}
#endif
