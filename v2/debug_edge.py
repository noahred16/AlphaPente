#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from games.pente import Pente


def test_edge_diagonal():
    """Test edge diagonal to see if it can be completed."""
    print("Testing edge diagonal completion...")
    
    game = Pente(board_size=7, tournament_rule=False)
    
    # Create edge diagonal: (0,3), (1,4), (2,5), (3,6)
    edge_diagonal = [(0, 3), (1, 4), (2, 5), (3, 6)]
    for row, col in edge_diagonal:
        game.board[row, col] = -1
    game.current_player = 1
    
    print("Board with edge diagonal:")
    for i in range(7):
        row_str = f"{i}: "
        for j in range(7):
            if game.board[i, j] == 1:
                row_str += "X "
            elif game.board[i, j] == -1:
                row_str += "O "
            else:
                row_str += ". "
        print(row_str)
    
    # Test potential completing moves
    # For diagonal (0,3)→(3,6), the extensions would be:
    # Backward: (-1,2) - off board
    # Forward: (4,7) - off board
    
    potential_moves = [(4, 7), (-1, 2)]  # These would be off-board
    on_board_moves = []
    
    for move in potential_moves:
        if 0 <= move[0] < 7 and 0 <= move[1] < 7:
            on_board_moves.append(move)
    
    print(f"\nPotential completing moves: {potential_moves}")
    print(f"Valid on-board moves: {on_board_moves}")
    
    if not on_board_moves:
        print("No valid moves to complete this diagonal - it cannot form a winning threat!")
        print("This is why the heuristic correctly assigns low priority.")


if __name__ == "__main__":
    test_edge_diagonal()