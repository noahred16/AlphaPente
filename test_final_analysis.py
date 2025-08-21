#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from games.pente import Pente
from players.mcts_player import MCTSPlayer
from mcts.move_heuristic import MoveHeuristic


def print_board(game):
    """Print the current board state."""
    print("Board:")
    for i in range(game.board_size):
        row_str = f"{i}: "
        for j in range(game.board_size):
            if game.board[i, j] == 1:
                row_str += "X "
            elif game.board[i, j] == -1:
                row_str += "O "
            else:
                row_str += ". "
        print(row_str)


def test_heuristic_evaluation():
    """Test the heuristic evaluation directly."""
    print("=== HEURISTIC EVALUATION TEST ===")
    
    game = Pente(board_size=7, tournament_rule=False)
    
    # Set up the problematic position
    mcts_stones = [(2, 1), (2, 2), (2, 3), (2, 4)]  # Can win at (2,5) or (2,0)
    for row, col in mcts_stones:
        game.board[row, col] = 1
        
    opponent_stones = [(5, 1), (5, 2), (5, 3), (5, 4)]  # Threat at (5,5) or (5,0)
    for row, col in opponent_stones:
        game.board[row, col] = -1
    
    game.current_player = 1
    print_board(game)
    
    # Test heuristic evaluation
    heuristic = MoveHeuristic(game)
    legal_moves = game.get_legal_moves()
    evaluated_moves = heuristic.evaluate_moves(legal_moves)
    
    print(f"\nTop 10 moves by heuristic score:")
    for i, (move, score) in enumerate(evaluated_moves[:10]):
        move_type = "UNKNOWN"
        if move in [(2, 0), (2, 5), (2, 6)]:
            move_type = "WIN"
        elif move in [(5, 0), (5, 5), (5, 6)]:
            move_type = "BLOCK"
        
        print(f"{i+1:2}. {move} = {score:3.0f} ({move_type})")
    
    # Test if heuristic correctly identifies critical moves
    critical_moves = [(2, 0), (2, 5), (5, 0), (5, 5)]
    print(f"\nCritical move evaluations:")
    for move in critical_moves:
        if move in game.get_legal_moves():
            score = heuristic._evaluate_move(move)
            print(f"  {move}: {score}")


def test_diagonal_scenarios():
    """Test MCTS performance on diagonal blocking scenarios."""
    print("\n=== DIAGONAL BLOCKING SCENARIOS ===")
    
    iterations = 100
    
    # Test 1: Diagonal winning position (\)
    print("\n1. Diagonal winning position (\\):")
    game1 = Pente(board_size=7, tournament_rule=False)
    diagonal_positions = [(1, 1), (2, 2), (3, 3), (4, 4)]
    for row, col in diagonal_positions:
        game1.board[row, col] = 1
    game1.current_player = 1
    
    print_board(game1)
    player1 = MCTSPlayer("MCTS", 1, iterations)
    move1 = player1.get_move(game1)
    
    winning_moves = [(0, 0), (5, 5), (6, 6)]
    print(f"MCTS chose: {move1} - {'WIN!' if move1 in winning_moves else 'Miss'}")
    
    # Test 2: Diagonal blocking position (\)
    print("\n2. Diagonal blocking position (\\):")
    game2 = Pente(board_size=7, tournament_rule=False)
    diagonal_positions = [(1, 1), (2, 2), (3, 3), (4, 4)]
    for row, col in diagonal_positions:
        game2.board[row, col] = -1
    game2.current_player = 1
    
    print_board(game2)
    player2 = MCTSPlayer("MCTS", 1, iterations)
    move2 = player2.get_move(game2)
    
    blocking_moves = [(0, 0), (5, 5), (6, 6)]
    print(f"MCTS chose: {move2} - {'BLOCK!' if move2 in blocking_moves else 'Miss'}")
    
    # Test 3: Diagonal winning position (/)
    print("\n3. Diagonal winning position (/):")
    game3 = Pente(board_size=7, tournament_rule=False)
    diagonal_positions = [(4, 1), (3, 2), (2, 3), (1, 4)]
    for row, col in diagonal_positions:
        game3.board[row, col] = 1
    game3.current_player = 1
    
    print_board(game3)
    player3 = MCTSPlayer("MCTS", 1, iterations)
    move3 = player3.get_move(game3)
    
    winning_moves = [(5, 0), (0, 5), (6, 0)]
    print(f"MCTS chose: {move3} - {'WIN!' if move3 in winning_moves else 'Miss'}")
    
    # Test 4: Diagonal blocking position (/)
    print("\n4. Diagonal blocking position (/):")
    game4 = Pente(board_size=7, tournament_rule=False)
    diagonal_positions = [(4, 1), (3, 2), (2, 3), (1, 4)]
    for row, col in diagonal_positions:
        game4.board[row, col] = -1
    game4.current_player = 1
    
    print_board(game4)
    player4 = MCTSPlayer("MCTS", 1, iterations)
    move4 = player4.get_move(game4)
    
    blocking_moves = [(5, 0), (0, 5), (6, 0)]
    print(f"MCTS chose: {move4} - {'BLOCK!' if move4 in blocking_moves else 'Miss'}")
    
    # Test 5: Complex diagonal scenario (win vs block)
    print("\n5. Complex diagonal scenario (win vs block):")
    game5 = Pente(board_size=7, tournament_rule=False)
    # MCTS diagonal winning opportunity (\)
    win_diagonal = [(1, 1), (2, 2), (3, 3), (4, 4)]
    for row, col in win_diagonal:
        game5.board[row, col] = 1
    # Opponent diagonal threat (/)
    block_diagonal = [(4, 2), (3, 3), (2, 4), (1, 5)]
    for row, col in block_diagonal:
        if game5.board[row, col] == 0:  # Don't overwrite existing stones
            game5.board[row, col] = -1
    game5.current_player = 1
    
    print_board(game5)
    player5 = MCTSPlayer("MCTS", 1, iterations)
    move5 = player5.get_move(game5)
    
    win_moves = [(0, 0), (5, 5), (6, 6)]
    block_moves = [(5, 1), (0, 6)]
    
    if move5 in win_moves:
        result = "WIN!"
    elif move5 in block_moves:
        result = "BLOCK!"
    else:
        result = "OTHER"
    
    print(f"MCTS chose: {move5} - {result}")
    
    # Test 6: Edge diagonal blocking
    print("\n6. Edge diagonal blocking:")
    game6 = Pente(board_size=7, tournament_rule=False)
    edge_diagonal = [(0, 3), (1, 4), (2, 5), (3, 6)]
    for row, col in edge_diagonal:
        game6.board[row, col] = -1
    game6.current_player = 1
    
    print_board(game6)
    player6 = MCTSPlayer("MCTS", 1, iterations)
    move6 = player6.get_move(game6)
    
    # Only one valid blocking move due to board edge
    blocking_moves = [(4, 7)]  # This would be off-board, so check valid moves
    valid_blocks = []
    for r in range(-1, 2):
        for c in range(-1, 2):
            test_pos = (3 + r, 6 + c)
            if (0 <= test_pos[0] < 7 and 0 <= test_pos[1] < 7 and 
                game6.board[test_pos[0], test_pos[1]] == 0):
                # Check if this completes the diagonal
                if test_pos[0] - 3 == test_pos[1] - 6:  # Same diagonal slope
                    valid_blocks.append(test_pos)
    
    # Since edge limits options, check if MCTS makes any reasonable defensive move
    print(f"MCTS chose: {move6} - {'BLOCK!' if move6 in [(4, 5), (5, 4)] else 'Miss'}")


def test_simple_vs_complex():
    """Compare MCTS performance on simple vs complex positions."""
    print("\n=== SIMPLE VS COMPLEX POSITION TEST ===")
    
    iterations = 100
    
    # Test 1: Simple winning position
    print("\n1. Simple winning position:")
    game1 = Pente(board_size=7, tournament_rule=False)
    for col in range(1, 5):
        game1.board[3, col] = 1
    game1.current_player = 1
    
    print_board(game1)
    player1 = MCTSPlayer("MCTS", 1, iterations)
    move1 = player1.get_move(game1)
    
    winning_moves = [(3, 0), (3, 5), (3, 6)]
    print(f"MCTS chose: {move1} - {'WIN!' if move1 in winning_moves else 'Miss'}")
    
    # Test 2: Simple blocking position
    print("\n2. Simple blocking position:")
    game2 = Pente(board_size=7, tournament_rule=False)
    for col in range(1, 5):
        game2.board[3, col] = -1
    game2.current_player = 1
    
    print_board(game2)
    player2 = MCTSPlayer("MCTS", 1, iterations)
    move2 = player2.get_move(game2)
    
    blocking_moves = [(3, 0), (3, 5), (3, 6)]
    print(f"MCTS chose: {move2} - {'BLOCK!' if move2 in blocking_moves else 'Miss'}")
    
    # Test 3: Complex position (win + block available)
    print("\n3. Complex position (win + block):")
    game3 = Pente(board_size=7, tournament_rule=False)
    # MCTS winning opportunity
    for col in range(1, 5):
        game3.board[2, col] = 1
    # Opponent threat
    for col in range(1, 5):
        game3.board[5, col] = -1
    game3.current_player = 1
    
    print_board(game3)
    player3 = MCTSPlayer("MCTS", 1, iterations)
    move3 = player3.get_move(game3)
    
    win_moves = [(2, 0), (2, 5), (2, 6)]
    block_moves = [(5, 0), (5, 5), (5, 6)]
    critical_moves = win_moves + block_moves
    
    if move3 in win_moves:
        result = "WIN!"
    elif move3 in block_moves:
        result = "BLOCK!"
    elif move3 in critical_moves:
        result = "CRITICAL!"
    else:
        result = "Miss"
    
    print(f"MCTS chose: {move3} - {result}")


if __name__ == "__main__":
    test_heuristic_evaluation()
    test_diagonal_scenarios()
    test_simple_vs_complex()