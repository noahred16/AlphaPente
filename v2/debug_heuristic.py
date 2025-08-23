#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from games.pente import Pente
from mcts.move_heuristic import MoveHeuristic

def debug_heuristic_test():
    """Debug the exact scenario from the failing test."""
    game = Pente(board_size=7, tournament_rule=False)
    
    # Set up the exact same scenario as the test
    # Critical threat - opponent about to win
    critical_threat = [(3, 1), (3, 2), (3, 3), (3, 4)]  # Block at (3,5) or (3,0)
    for row, col in critical_threat:
        game.board[row, col] = -1
        
    # High priority - MCTS can create open three
    mcts_stones = [(5, 2), (5, 3)]  # Can create open three at (5,1) or (5,4)
    for row, col in mcts_stones:
        game.board[row, col] = 1
        
    # Medium priority - capture opportunity
    game.board[1, 1] = 1  # MCTS stone
    game.board[1, 2] = -1  # Opponent stones
    game.board[1, 3] = -1
    # Can capture at (1,4)
    
    game.current_player = 1
    
    print("Board state:")
    print(game.board)
    print()
    
    # Get all legal moves and evaluate them
    legal_moves = game.get_legal_moves()
    heuristic = MoveHeuristic(game)
    scored_moves = heuristic.evaluate_moves(legal_moves)
    
    print("Top 15 moves with heuristic scores:")
    for i, (move, score) in enumerate(scored_moves[:15]):
        category = "OTHER"
        if move in [(3, 5), (3, 0)]:
            category = "CRITICAL_BLOCK"
        elif move in [(5, 1), (5, 4)]:
            category = "OPEN_THREE"
        elif move == (1, 4):
            category = "CAPTURE"
            
        print(f"{i+1:2d}. {move} - Score: {score:6.1f} - Category: {category}")
    
    print(f"\nTotal legal moves: {len(legal_moves)}")
    
    # Test specific moves we care about
    test_moves = {
        (3, 5): "CRITICAL_BLOCK",
        (3, 0): "CRITICAL_BLOCK", 
        (5, 1): "OPEN_THREE",
        (5, 4): "OPEN_THREE",
        (1, 4): "CAPTURE"
    }
    
    print("\nDetailed analysis of key moves:")
    for move, expected_category in test_moves.items():
        if move in [m for m, s in scored_moves]:
            score = next(s for m, s in scored_moves if m == move)
            print(f"{move} ({expected_category}): Score = {score}")
            
            # Test individual conditions
            print(f"  _is_winning_move: {heuristic._is_winning_move(move, 1)}")
            print(f"  _is_blocking_win: {heuristic._is_blocking_win(move, -1)}")
            print(f"  _creates_capture: {heuristic._creates_capture(move, 1)}")
            print(f"  _creates_open_three: {heuristic._creates_open_three(move, 1)}")
            print(f"  _blocks_open_three: {heuristic._blocks_open_three(move, -1)}")
            print(f"  _is_near_cluster: {heuristic._is_near_cluster(move)}")
        else:
            print(f"{move} ({expected_category}): NOT IN LEGAL MOVES")

if __name__ == "__main__":
    debug_heuristic_test()