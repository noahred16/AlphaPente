#include "core/game_state.hpp"
#include "core/move_generator.hpp"
#include "core/distance_rings.hpp"
#include "mcts/mcts_engine.hpp"
#include <iostream>
#include <iomanip>
#include <memory>

using namespace core;
using namespace mcts;

std::unique_ptr<GameState> create_test_state() {
    auto state = std::make_unique<GameState>();
    // Set up a four-in-a-row threat for the human player (engine must block)
    // Human stones: J10, K10, L10, M10 (horizontal four threat)
    // Cover one side with engine stones (I10)
    // expect engine to block the other side
    state->make_move(9, 9);   // J10 (Human)
    state->make_move(9, 8);   // I10 (Engine - covers one side)
    state->make_move(9, 10);  // K10 (Human)
    state->make_move(6, 6);   // G7 (Engine - irrelevant move)
    state->make_move(9, 11);  // L10 (Human)
    state->make_move(7, 7);   // H8 (Engine - irrelevant move)
    state->make_move(9, 12);  // M10 (Human) - creates four-in-a-row threat
    return state;
}

void print_board(const GameState& state) {
    std::cout << "\nBoard state with four threat (J10-K10-L10-M10):\n";
    std::cout << "    A B C D E F G H I J K L M N O P Q R S\n";
    for (int row = 0; row < 19; row++) {
        printf("%2d  ", 19-row);
        for (int col = 0; col < 19; col++) {
            int stone = state.get_stone(row, col);
            if (stone == 1) std::cout << "X ";
            else if (stone == -1) std::cout << "O ";
            else std::cout << ". ";
        }
        printf("  %d\n", 19-row);
    }
    std::cout << "    A B C D E F G H I J K L M N O P Q R S\n";
}

void analyze_node_expansion(const MCTSNode* node, const std::string& node_name, int depth = 0) {
    if (!node) return;

    std::string indent(depth * 2, ' ');
    const auto& move = node->get_move();
    char move_col = 'A' + move.col;
    int move_row = 19 - move.row;

    std::cout << indent << "Node " << node_name;
    if (move.row != -1) {
        std::cout << " (move: " << move_col << move_row << ")";
    }
    std::cout << ":\n";
    std::cout << indent << "  Visits: " << node->get_visits() << "\n";
    std::cout << indent << "  Wins: " << node->get_wins() << "\n";
    std::cout << indent << "  Win rate: " << std::fixed << std::setprecision(2) << node->get_win_rate() << "\n";
    std::cout << indent << "  Is leaf: " << (node->is_leaf() ? "true" : "false") << "\n";
    std::cout << indent << "  Is fully expanded: " << (node->is_fully_expanded() ? "true" : "false") << "\n";
    std::cout << indent << "  Child count: " << node->child_count() << "\n";

    auto children = node->get_top_children(-1);
    std::cout << indent << "  Children via get_top_children(): " << children.size() << "\n";

    if (children.size() != node->child_count()) {
        std::cout << indent << "  WARNING: Child count mismatch!\n";
    }
}

int main() {
    std::cout << "=== MCTS Tree Debug Script ===\n";

    // Create test state
    auto state = create_test_state();
    print_board(*state);

    // Create move generator and engine
    auto distance_rings = std::make_unique<DistanceRings>();
    auto move_generator = std::make_unique<MoveGenerator>(*distance_rings);
    auto engine = std::make_unique<MCTSEngine>(*state, *move_generator);

    std::cout << "\n=== Running MCTS Search ===\n";
    Position engine_move = engine->search(3000, 6000.0);

    if (engine_move.row == -1 || engine_move.col == -1) {
        std::cout << "ERROR: Engine returned invalid move!\n";
        return 1;
    }

    char col_char = 'A' + engine_move.col;
    int display_row = 19 - engine_move.row;
    std::cout << "Engine played: " << col_char << display_row << "\n";

    std::cout << "\n=== Root Node Analysis ===\n";
    analyze_node_expansion(engine->get_root(), "ROOT");

    std::cout << "\n=== Top Children by Visits ===\n";
    auto top_children = engine->get_root()->get_top_children(-1);
    std::cout << "Total children: " << top_children.size() << "\n";

    for (size_t i = 0; i < std::min(size_t(10), top_children.size()); i++) {
        const auto* child = top_children[i];
        const auto& move = child->get_move();
        char move_col = 'A' + move.col;
        int move_row = 19 - move.row;
        double win_rate = child->get_win_rate();
        std::cout << "  " << (i+1) << ". " << move_col << move_row
                    << " (visits: " << child->get_visits()
                    << ", win rate: " << std::fixed << std::setprecision(2) << win_rate << ")\n";
    }

    if (!top_children.empty()) {
        std::cout << "\n=== Analyzing Top Move ===\n";
        const auto* best_child = top_children[0];
        const auto& move = best_child->get_move();
        char move_col = 'A' + move.col;
        int move_row = 19 - move.row;

        std::cout << "Top choice: " << move_col << move_row << "\n";
        analyze_node_expansion(best_child, "BEST_CHILD");

        auto best_child_children = best_child->get_top_children(-1);
        std::cout << "\nBest child's children (" << best_child_children.size() << "):\n";

        if (best_child_children.size() == 0) {
            std::cout << "*** PROBLEM: Top choice move has no children! ***\n";
            std::cout << "This indicates the node was visited but never expanded.\n";
            std::cout << "Possible causes:\n";
            std::cout << "1. Node represents a terminal game state\n";
            std::cout << "2. Expansion failed due to no legal moves\n";
            std::cout << "3. Bug in expansion logic\n";
        } else {
            for (size_t i = 0; i < std::min(size_t(5), best_child_children.size()); i++) {
                const auto* child = best_child_children[i];
                const auto& move = child->get_move();
                char move_col = 'A' + move.col;
                int move_row = 19 - move.row;
                double win_rate = child->get_win_rate();
                std::cout << "     " << (i+1) << ". " << move_col << move_row
                            << " (visits: " << child->get_visits()
                            << ", win rate: " << std::fixed << std::setprecision(2) << win_rate << ")\n";
            }
        }
    }

    std::cout << "\n=== Blocking Move Analysis ===\n";
    Position expected_block = {9, 13}; // N10
    char expected_col = 'A' + expected_block.col;
    int expected_row = 19 - expected_block.row;

    bool found_blocking_move = false;
    for (const auto* child : top_children) {
        const auto& move = child->get_move();
        if (move.row == expected_block.row && move.col == expected_block.col) {
            char move_col = 'A' + move.col;
            int move_row = 19 - move.row;
            double win_rate = child->get_win_rate();
            std::cout << "Found blocking move " << move_col << move_row
                        << " (visits: " << child->get_visits()
                        << ", win rate: " << std::fixed << std::setprecision(2) << win_rate << ")\n";
            found_blocking_move = true;
            break;
        }
    }

    if (!found_blocking_move) {
        std::cout << "Expected blocking move " << expected_col << expected_row << " was NOT explored!\n";
    }

    bool is_blocking = (engine_move.row == expected_block.row && engine_move.col == expected_block.col);
    std::cout << "Engine chose blocking move: " << (is_blocking ? "YES" : "NO") << "\n";

    return 0;
}