"""
GameState class with single state and efficient undo functionality.
Critical for high-performance MCTS implementation.
"""

from typing import List, Tuple, Optional
from .bitboard import BitBoard
from .move_delta import MoveDelta

class GameState:
    """
    Pente game state using single BitBoard with efficient make/undo operations.
    
    This is the core of the high-performance MCTS implementation:
    - Never copies state (eliminates thousands of deep copies per second)
    - Uses move deltas for perfect undo operations
    - Orders of magnitude faster than state copying approach
    """
    
    def __init__(self, board_size: int = 19, captures_to_win: int = 5, tournament_rule: bool = True):
        self.board_size = board_size
        self.captures_to_win = captures_to_win
        self.tournament_rule = tournament_rule
        
        # Single BitBoard instance - never copied!
        self.board = BitBoard()
        
        # Game state
        self.current_player = 1
        self.move_history: List[MoveDelta] = []
        self.captures = {1: 0, -1: 0}  # Number of pairs captured
        
        # Performance tracking
        self.move_count = 0
    
    def make_move(self, position: Tuple[int, int]) -> MoveDelta:
        """
        Make a move and return delta for undo.
        This is the critical path for MCTS performance.
        """
        row, col = position
        
        if not self.board.is_empty(row, col):
            raise ValueError(f"Position {position} is already occupied")
        
        # Place the stone
        self.board.set_stone(row, col, self.current_player)
        
        # Check for captures
        captured_positions = self._check_captures(row, col, self.current_player)
        captured_player = -self.current_player if captured_positions else 0
        capture_count = len(captured_positions) // 2  # Captures are in pairs
        
        # Update capture count
        if capture_count > 0:
            self.captures[self.current_player] += capture_count
        
        # Create move delta
        delta = MoveDelta(
            position=position,
            player=self.current_player,
            captured_positions=captured_positions,
            captured_player=captured_player,
            capture_count=capture_count
        )
        
        # Update state
        self.move_history.append(delta)
        self.move_count += 1
        self.current_player *= -1
        
        return delta
    
    def undo_move(self) -> Optional[MoveDelta]:
        """
        Undo the last move using the move delta.
        Returns the delta that was undone, or None if no moves to undo.
        """
        if not self.move_history:
            return None
        
        delta = self.move_history.pop()
        
        # Restore game state
        self.current_player = delta.player
        self.move_count -= 1
        
        # Remove the stone that was placed
        self.board.remove_stone(*delta.position)
        
        # Restore captured stones
        if delta.has_captures():
            for pos in delta.captured_positions:
                self.board.set_stone(*pos, delta.captured_player)
            self.captures[delta.player] -= delta.capture_count
        
        return delta
    
    def _check_captures(self, row: int, col: int, player: int) -> List[Tuple[int, int]]:
        """
        Check for captures after placing a stone.
        Returns list of captured stone positions.
        """
        directions = [(0, 1), (1, 0), (1, 1), (1, -1), (0, -1), (-1, 0), (-1, -1), (-1, 1)]
        captured_positions = []
        opponent = -player
        
        for dr, dc in directions:
            # Look for pattern: player -> opponent -> opponent -> player
            positions_to_check = []
            
            # Check first opponent stone
            r1, c1 = row + dr, col + dc
            if (0 <= r1 < self.board_size and 0 <= c1 < self.board_size and 
                self.board.get_stone(r1, c1) == opponent):
                positions_to_check.append((r1, c1))
                
                # Check second opponent stone
                r2, c2 = r1 + dr, c1 + dc
                if (0 <= r2 < self.board_size and 0 <= c2 < self.board_size and 
                    self.board.get_stone(r2, c2) == opponent):
                    positions_to_check.append((r2, c2))
                    
                    # Check for closing player stone
                    r3, c3 = r2 + dr, c2 + dc
                    if (0 <= r3 < self.board_size and 0 <= c3 < self.board_size and 
                        self.board.get_stone(r3, c3) == player):
                        # Capture found! Remove the two opponent stones
                        for cap_pos in positions_to_check:
                            self.board.remove_stone(*cap_pos)
                            captured_positions.append(cap_pos)
        
        return captured_positions
    
    def get_legal_moves(self) -> List[Tuple[int, int]]:
        """Get all legal moves for the current player."""
        moves = []
        
        # Tournament rule: Player -1's first move (second move of game) must be ≥3 from center
        if (self.tournament_rule and self.move_count == 1 and self.current_player == -1):
            center = self.board_size // 2
            for row in range(self.board_size):
                for col in range(self.board_size):
                    if self.board.is_empty(row, col):
                        distance = max(abs(row - center), abs(col - center))
                        if distance >= 3:
                            moves.append((row, col))
        else:
            # Normal case: all empty positions
            for row in range(self.board_size):
                for col in range(self.board_size):
                    if self.board.is_empty(row, col):
                        moves.append((row, col))
        
        return moves
    
    def is_terminal(self) -> bool:
        """Check if the game has ended."""
        return self.get_winner() is not None or len(self.get_legal_moves()) == 0
    
    def get_winner(self) -> Optional[int]:
        """
        Get the winner of the game.
        Returns 1, -1, or None (for ongoing game).
        """
        # Check capture win
        for player in [1, -1]:
            if self.captures[player] >= self.captures_to_win:
                return player
        
        # Check five-in-a-row
        return self._check_five_in_row()
    
    def _check_five_in_row(self) -> Optional[int]:
        """Check for five stones in a row."""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for row in range(self.board_size):
            for col in range(self.board_size):
                stone = self.board.get_stone(row, col)
                if stone != 0:
                    for dr, dc in directions:
                        if self._count_line(row, col, dr, dc, stone) >= 5:
                            return stone
        
        return None
    
    def _count_line(self, start_row: int, start_col: int, dr: int, dc: int, player: int) -> int:
        """Count consecutive stones in a direction."""
        count = 1  # Count the starting stone
        
        # Count forward
        r, c = start_row + dr, start_col + dc
        while (0 <= r < self.board_size and 0 <= c < self.board_size and 
               self.board.get_stone(r, c) == player):
            count += 1
            r += dr
            c += dc
        
        # Count backward
        r, c = start_row - dr, start_col - dc
        while (0 <= r < self.board_size and 0 <= c < self.board_size and 
               self.board.get_stone(r, c) == player):
            count += 1
            r -= dr
            c -= dc
        
        return count
    
    def get_board_state(self) -> List[Tuple[int, int, int]]:
        """Get current board state as list of (row, col, player) tuples."""
        return self.board.get_all_stones()
    
    def __str__(self) -> str:
        """String representation of the game state."""
        lines = [f"Move {self.move_count}, Player {self.current_player} to play"]
        lines.append(f"Captures - Player 1: {self.captures[1]}, Player -1: {self.captures[-1]}")
        lines.append(str(self.board))
        return '\n'.join(lines)