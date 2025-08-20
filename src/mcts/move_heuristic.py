from typing import List, Tuple, Dict
import numpy as np
from games.pente import Pente

class MoveHeuristic:
    """Heuristic evaluation for Pente moves with priority-based ordering."""
    
    WEIGHTS = {
        'critical': 100,      # Win now, block opponent's win
        'very_high': 50,      # Capture stone pair, create open 4
        'high': 20,           # Create open 3, block opponent open 3
        'medium': 5,          # Move near cluster (distance ≤ 2)
        'low': 1              # Anything else
    }
    
    def __init__(self, game: Pente):
        self.game = game
        self.board_size = game.board_size
    
    def evaluate_moves(self, moves: List[Tuple[int, int]]) -> List[Tuple[Tuple[int, int], float]]:
        """Evaluate all moves and return them sorted by priority."""
        move_scores = []
        
        for move in moves:
            score = self._evaluate_move(move)
            move_scores.append((move, score))
        
        return sorted(move_scores, key=lambda x: x[1], reverse=True)
    
    def _evaluate_move(self, move: Tuple[int, int]) -> float:
        """Evaluate a single move and return its heuristic score."""
        row, col = move
        player = self.game.current_player
        opponent = -player
        
        # Check for critical moves (win/block win)
        if self._is_winning_move(move, player):
            return self.WEIGHTS['critical']
        
        if self._is_blocking_win(move, opponent):
            return self.WEIGHTS['critical']
        
        # Check for very high priority moves
        if self._creates_capture(move, player):
            return self.WEIGHTS['very_high']
        
        if self._creates_open_four(move, player):
            return self.WEIGHTS['very_high']
        
        # Check for high priority moves
        if self._creates_open_three(move, player):
            return self.WEIGHTS['high']
        
        if self._blocks_open_three(move, opponent):
            return self.WEIGHTS['high']
        
        # Check for medium priority (near cluster)
        if self._is_near_cluster(move):
            return self.WEIGHTS['medium']
        
        # Default low priority
        return self.WEIGHTS['low']
    
    def _is_winning_move(self, move: Tuple[int, int], player: int) -> bool:
        """Check if move creates a winning condition."""
        game_copy = self.game.clone()
        game_copy.make_move(move)
        return game_copy.get_winner() == player
    
    def _is_blocking_win(self, move: Tuple[int, int], opponent: int) -> bool:
        """Check if move blocks opponent's winning move."""
        # Temporarily switch to opponent
        original_player = self.game.current_player
        self.game.current_player = opponent
        
        is_blocking = self._is_winning_move(move, opponent)
        
        # Restore original player
        self.game.current_player = original_player
        return is_blocking
    
    def _creates_capture(self, move: Tuple[int, int], player: int) -> bool:
        """Check if move creates a capture opportunity."""
        row, col = move
        directions = [(0, 1), (1, 0), (1, 1), (1, -1), (0, -1), (-1, 0), (-1, -1), (-1, 1)]
        
        for dr, dc in directions:
            r1, c1 = row + dr, col + dc
            r2, c2 = row + 2*dr, col + 2*dc
            r3, c3 = row + 3*dr, col + 3*dc
            
            if (0 <= r1 < self.board_size and 0 <= c1 < self.board_size and
                0 <= r2 < self.board_size and 0 <= c2 < self.board_size and
                0 <= r3 < self.board_size and 0 <= c3 < self.board_size):
                
                if (self.game.board[r1, c1] == -player and 
                    self.game.board[r2, c2] == -player and 
                    self.game.board[r3, c3] == player):
                    return True
        
        return False
    
    def _creates_open_four(self, move: Tuple[int, int], player: int) -> bool:
        """Check if move creates an open four (4 in a row with open ends)."""
        return self._creates_threat(move, player, 4, True)
    
    def _creates_open_three(self, move: Tuple[int, int], player: int) -> bool:
        """Check if move creates an open three (3 in a row with open ends)."""
        return self._creates_threat(move, player, 3, True)
    
    def _blocks_open_three(self, move: Tuple[int, int], opponent: int) -> bool:
        """Check if move blocks opponent's open three."""
        row, col = move
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dr, dc in directions:
            # Check if placing here disrupts opponent's potential open three
            count = 0
            for offset in range(-4, 5):
                r, c = row + offset * dr, col + offset * dc
                if 0 <= r < self.board_size and 0 <= c < self.board_size:
                    if self.game.board[r, c] == opponent:
                        count += 1
                    elif self.game.board[r, c] == -opponent:
                        count = 0
            
            if count >= 2:  # Disrupts a potential threat
                return True
        
        return False
    
    def _creates_threat(self, move: Tuple[int, int], player: int, length: int, open_ends: bool) -> bool:
        """Check if move creates a threat of specified length."""
        row, col = move
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dr, dc in directions:
            count = 1  # Count the placed stone
            open_count = 0
            
            # Count in positive direction
            r, c = row + dr, col + dc
            while (0 <= r < self.board_size and 0 <= c < self.board_size and 
                   self.game.board[r, c] == player):
                count += 1
                r += dr
                c += dc
            
            # Check if end is open
            if (0 <= r < self.board_size and 0 <= c < self.board_size and 
                self.game.board[r, c] == 0):
                open_count += 1
            
            # Count in negative direction
            r, c = row - dr, col - dc
            while (0 <= r < self.board_size and 0 <= c < self.board_size and 
                   self.game.board[r, c] == player):
                count += 1
                r -= dr
                c -= dc
            
            # Check if other end is open
            if (0 <= r < self.board_size and 0 <= c < self.board_size and 
                self.game.board[r, c] == 0):
                open_count += 1
            
            if count >= length and (not open_ends or open_count >= 2):
                return True
        
        return False
    
    def _is_near_cluster(self, move: Tuple[int, int], max_distance: int = 2) -> bool:
        """Check if move is within max_distance of existing stones."""
        row, col = move
        
        for i in range(max(0, row - max_distance), min(self.board_size, row + max_distance + 1)):
            for j in range(max(0, col - max_distance), min(self.board_size, col + max_distance + 1)):
                if self.game.board[i, j] != 0:
                    distance = max(abs(i - row), abs(j - col))
                    if distance <= max_distance:
                        return True
        
        return False