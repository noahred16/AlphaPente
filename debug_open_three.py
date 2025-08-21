#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from games.pente import Pente
from mcts.move_heuristic import MoveHeuristic

def debug_open_three():
    """Debug open three detection specifically."""
    game = Pente(board_size=7, tournament_rule=False)
    
    # Set up stones at (5,2) and (5,3)
    game.board[5, 2] = 1
    game.board[5, 3] = 1
    game.current_player = 1
    
    print("Board with two stones at (5,2) and (5,3):")
    print(game.board)
    print()
    
    heuristic = MoveHeuristic(game)
    
    # Test moves (5,1) and (5,4) - should create open three
    test_moves = [(5, 1), (5, 4)]
    
    for move in test_moves:
        print(f"Testing move {move}:")
        print(f"  _creates_open_three: {heuristic._creates_open_three(move, 1)}")
        print(f"  _creates_threat(3, True): {heuristic._creates_threat(move, 1, 3, True)}")
        
        # Manually check the logic
        row, col = move
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dr, dc in directions:
            print(f"  Direction ({dr}, {dc}):")
            count = 1  # Count the placed stone
            open_count = 0
            
            # Count in positive direction
            r, c = row + dr, col + dc
            pos_count = 0
            while (0 <= r < 7 and 0 <= c < 7 and game.board[r, c] == 1):
                pos_count += 1
                count += 1
                r += dr
                c += dc
            
            # Check if end is open
            pos_open = (0 <= r < 7 and 0 <= c < 7 and game.board[r, c] == 0)
            if pos_open:
                open_count += 1
            
            print(f"    Positive direction: found {pos_count} stones, open: {pos_open}")
            
            # Count in negative direction  
            r, c = row - dr, col - dc
            neg_count = 0
            while (0 <= r < 7 and 0 <= c < 7 and game.board[r, c] == 1):
                neg_count += 1
                count += 1
                r -= dr
                c -= dc
            
            # Check if other end is open
            neg_open = (0 <= r < 7 and 0 <= c < 7 and game.board[r, c] == 0)
            if neg_open:
                open_count += 1
                
            print(f"    Negative direction: found {neg_count} stones, open: {neg_open}")
            print(f"    Total count: {count}, open ends: {open_count}")
            print(f"    Would create open three: {count >= 3 and open_count >= 2}")
        
        print()

if __name__ == "__main__":
    debug_open_three()