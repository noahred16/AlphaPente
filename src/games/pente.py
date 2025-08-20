from typing import List, Tuple, Optional
import numpy as np
import copy
from .base_game import BaseGame

class Pente(BaseGame):
    """Pente game implementation with configurable rules."""
    
    def __init__(self, board_size: int = 19, captures_to_win: int = 5, tournament_rule: bool = True):
        super().__init__(board_size)
        self.captures_to_win = captures_to_win
        self.tournament_rule = tournament_rule
        self.move_count = 0
        
    def get_legal_moves(self) -> List[Tuple[int, int]]:
        """Return list of legal moves."""
        moves = []
        center = self.board_size // 2
        
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i, j] == 0:
                    if self.tournament_rule and self.move_count == 1:
                        distance = max(abs(i - center), abs(j - center))
                        if distance >= 3:
                            moves.append((i, j))
                    else:
                        moves.append((i, j))
        
        return moves
    
    def make_move(self, move: Tuple[int, int]) -> None:
        """Execute a move on the board."""
        row, col = move
        if self.board[row, col] != 0:
            raise ValueError("Invalid move: position already occupied")
        
        self.board[row, col] = self.current_player
        self.move_history.append(move)
        self.move_count += 1
        
        captured = self._check_captures(row, col)
        self.captures[self.current_player] += captured
        
        self.current_player *= -1
    
    def _check_captures(self, row: int, col: int) -> int:
        """Check for captures after placing a stone."""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1), (0, -1), (-1, 0), (-1, -1), (-1, 1)]
        total_captured = 0
        
        for dr, dc in directions:
            captured_stones = []
            
            r, c = row + dr, col + dc
            if (0 <= r < self.board_size and 0 <= c < self.board_size and 
                self.board[r, c] == -self.current_player):
                captured_stones.append((r, c))
                
                r, c = r + dr, c + dc
                if (0 <= r < self.board_size and 0 <= c < self.board_size and 
                    self.board[r, c] == -self.current_player):
                    captured_stones.append((r, c))
                    
                    r, c = r + dr, c + dc
                    if (0 <= r < self.board_size and 0 <= c < self.board_size and 
                        self.board[r, c] == self.current_player):
                        for cap_r, cap_c in captured_stones:
                            self.board[cap_r, cap_c] = 0
                        total_captured += len(captured_stones)
        
        return total_captured
    
    def undo_move(self) -> None:
        """Undo the last move."""
        if not self.move_history:
            return
        
        last_move = self.move_history.pop()
        row, col = last_move
        self.move_count -= 1
        
        previous_player = -self.current_player
        self.board[row, col] = 0
        
        board_before = self._restore_captured_stones(row, col, previous_player)
        
        self.current_player = previous_player
    
    def _restore_captured_stones(self, row: int, col: int, player: int) -> None:
        """Restore stones that were captured by the last move."""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1), (0, -1), (-1, 0), (-1, -1), (-1, 1)]
        
        for dr, dc in directions:
            r1, c1 = row + dr, col + dc
            r2, c2 = row + 2*dr, col + 2*dc
            r3, c3 = row + 3*dr, col + 3*dc
            
            if (0 <= r1 < self.board_size and 0 <= c1 < self.board_size and
                0 <= r2 < self.board_size and 0 <= c2 < self.board_size and
                0 <= r3 < self.board_size and 0 <= c3 < self.board_size):
                
                if (self.board[r1, c1] == 0 and self.board[r2, c2] == 0 and 
                    self.board[r3, c3] == player):
                    self.board[r1, c1] = -player
                    self.board[r2, c2] = -player
                    self.captures[player] -= 2
    
    def is_terminal(self) -> bool:
        """Check if game has ended."""
        if self.get_winner() is not None:
            return True
        return len(self.get_legal_moves()) == 0
    
    def get_winner(self) -> Optional[int]:
        """Return winner (1, -1) or None."""
        for player in [1, -1]:
            if self.captures[player] >= self.captures_to_win:
                return player
        
        winner = self._check_five_in_row()
        return winner
    
    def _check_five_in_row(self) -> Optional[int]:
        """Check for five in a row."""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i, j] != 0:
                    player = self.board[i, j]
                    
                    for dr, dc in directions:
                        count = 1
                        
                        r, c = i + dr, j + dc
                        while (0 <= r < self.board_size and 0 <= c < self.board_size and 
                               self.board[r, c] == player):
                            count += 1
                            r += dr
                            c += dc
                        
                        r, c = i - dr, j - dc
                        while (0 <= r < self.board_size and 0 <= c < self.board_size and 
                               self.board[r, c] == player):
                            count += 1
                            r -= dr
                            c -= dc
                        
                        if count >= 5:
                            return player
        
        return None
    
    def clone(self) -> 'Pente':
        """Create a deep copy of the game state."""
        new_game = Pente(self.board_size, self.captures_to_win, self.tournament_rule)
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.captures = self.captures.copy()
        new_game.move_history = self.move_history.copy()
        new_game.move_count = self.move_count
        return new_game