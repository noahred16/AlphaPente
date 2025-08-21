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
        
        # Initialize legal moves cache with all empty positions
        self._legal_moves_cache = set()
        for i in range(board_size):
            for j in range(board_size):
                self._legal_moves_cache.add((i, j))
        
        # Cache for tournament rule constrained moves (computed lazily)
        self._tournament_legal_moves_cache = None
        self._tournament_cache_valid = False
        
    def get_legal_moves(self) -> List[Tuple[int, int]]:
        """Return list of legal moves using cached results."""
        # Verify cache consistency and auto-sync if needed (for robustness with tests)
        self._verify_cache_consistency()
        
        # Tournament rule: Player 1's second move must be ≥3 from center
        if self.tournament_rule and self.move_count == 2 and self.current_player == 1:
            if not self._tournament_cache_valid:
                self._update_tournament_cache()
            return list(self._tournament_legal_moves_cache)
        else:
            return list(self._legal_moves_cache)
    
    def _verify_cache_consistency(self):
        """Verify cache is consistent with board state and auto-sync if needed."""
        # Quick check: if cache size seems reasonable, skip expensive verification
        expected_size = self.board_size * self.board_size - np.count_nonzero(self.board)
        if len(self._legal_moves_cache) == expected_size:
            return  # Cache size matches, likely consistent
            
        # Full verification when cache size is suspicious
        actual_legal = set()
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i, j] == 0:
                    actual_legal.add((i, j))
        
        if actual_legal != self._legal_moves_cache:
            # Silently auto-sync for robustness
            self._legal_moves_cache = actual_legal
            self._tournament_cache_valid = False
    
    def _update_tournament_cache(self):
        """Update tournament rule cache for player 1's second move."""
        center = self.board_size // 2
        self._tournament_legal_moves_cache = set()
        
        for move in self._legal_moves_cache:
            i, j = move
            distance = max(abs(i - center), abs(j - center))
            if distance >= 3:
                self._tournament_legal_moves_cache.add(move)
        
        self._tournament_cache_valid = True
    
    def make_move(self, move: Tuple[int, int]) -> None:
        """Execute a move on the board."""
        row, col = move
        if self.board[row, col] != 0:
            raise ValueError("Invalid move: position already occupied")
        
        self.board[row, col] = self.current_player
        self.move_history.append(move)
        self.move_count += 1
        
        # Update legal moves cache - remove the occupied position
        self._legal_moves_cache.discard((row, col))
        self._tournament_cache_valid = False  # Invalidate tournament cache
        
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
                            # Add captured positions back to legal moves cache
                            self._legal_moves_cache.add((cap_r, cap_c))
                        total_captured += len(captured_stones)
                        self._tournament_cache_valid = False  # Invalidate tournament cache
        
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
        
        # Add the undone move position back to legal moves cache
        self._legal_moves_cache.add((row, col))
        self._tournament_cache_valid = False  # Invalidate tournament cache
        
        self._restore_captured_stones(row, col, previous_player)
        
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
                    # Remove restored stones from legal moves cache
                    self._legal_moves_cache.discard((r1, c1))
                    self._legal_moves_cache.discard((r2, c2))
                    self.captures[player] -= 2
                    self._tournament_cache_valid = False  # Invalidate tournament cache
    
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
        
        # Copy legal moves cache
        new_game._legal_moves_cache = self._legal_moves_cache.copy()
        new_game._tournament_cache_valid = self._tournament_cache_valid
        if self._tournament_legal_moves_cache is not None:
            new_game._tournament_legal_moves_cache = self._tournament_legal_moves_cache.copy()
        
        return new_game
    
    def _sync_legal_moves_cache(self):
        """Synchronize legal moves cache with current board state.
        
        This should be called if the board is modified directly without using make_move().
        Used primarily for testing scenarios where board positions are set manually.
        """
        self._legal_moves_cache.clear()
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i, j] == 0:
                    self._legal_moves_cache.add((i, j))
        
        # Invalidate tournament cache since board changed
        self._tournament_cache_valid = False