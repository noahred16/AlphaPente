#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from games.pente import Pente
from players.mcts_player import MCTSPlayer
from mcts.move_heuristic import MoveHeuristic

def debug_mcts_priority():
    """Debug MCTS move selection with the test scenario."""
    game = Pente(board_size=7, tournament_rule=False)
    
    # Set up the exact same scenario as the test
    critical_threat = [(3, 1), (3, 2), (3, 3), (3, 4)]
    for row, col in critical_threat:
        game.board[row, col] = -1
        
    mcts_stones = [(5, 2), (5, 3)]
    for row, col in mcts_stones:
        game.board[row, col] = 1
        
    game.board[1, 1] = 1
    game.board[1, 2] = -1
    game.board[1, 3] = -1
    
    game.current_player = 1
    
    print("Board state:")
    print(game.board)
    print()
    
    # Create MCTS player with detailed statistics
    player = MCTSPlayer(name="MCTS Test", player_id=1, max_iterations=50)
    
    # Get move statistics
    stats = player.get_move_statistics(game)
    
    print("MCTS Move Statistics (visits, win_rate from root perspective):")
    print("Move      Visits  WinRate  Category")
    print("-" * 40)
    
    for move, visits, win_rate in stats[:20]:  # Top 20 moves
        category = "OTHER"
        if move in [(3, 5), (3, 0)]:
            category = "CRITICAL_BLOCK"
        elif move in [(5, 1), (5, 4)]:
            category = "OPEN_THREE"
        elif move == (1, 4):
            category = "CAPTURE"
            
        print(f"{str(move):<9} {visits:>6} {win_rate:>8.3f}  {category}")
    
    print()
    
    # Also show heuristic ordering for comparison
    legal_moves = game.get_legal_moves()
    heuristic = MoveHeuristic(game)
    scored_moves = heuristic.evaluate_moves(legal_moves)
    
    print("Heuristic ordering (top 10):")
    print("Move      Score    Category")
    print("-" * 30)
    for move, score in scored_moves[:10]:
        category = "OTHER"
        if move in [(3, 5), (3, 0)]:
            category = "CRITICAL_BLOCK"
        elif move in [(5, 1), (5, 4)]:
            category = "OPEN_THREE"
        elif move == (1, 4):
            category = "CAPTURE"
            
        print(f"{str(move):<9} {score:>6.1f}    {category}")

if __name__ == "__main__":
    debug_mcts_priority()