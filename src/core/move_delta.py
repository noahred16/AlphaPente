"""
MoveDelta class for efficient undo operations.
Stores minimal information needed to reverse any move.
"""

from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class MoveDelta:
    """
    Stores the information needed to undo a move.
    
    This is critical for the single-state MCTS implementation where
    we make/undo moves on a single game state rather than copying states.
    """
    position: Tuple[int, int]  # Where the stone was placed
    player: int  # Who played the move (1 or -1)
    captured_positions: List[Tuple[int, int]]  # Positions of captured stones
    captured_player: int  # Which player's stones were captured (opposite of player)
    capture_count: int  # Number of pairs captured (for score tracking)
    
    def __init__(self, position: Tuple[int, int], player: int, 
                 captured_positions: List[Tuple[int, int]] = None,
                 captured_player: int = 0, capture_count: int = 0):
        self.position = position
        self.player = player
        self.captured_positions = list(captured_positions) if captured_positions else []
        self.captured_player = captured_player
        self.capture_count = capture_count
    
    def has_captures(self) -> bool:
        """Check if this move captured any stones."""
        return len(self.captured_positions) > 0
    
    def __repr__(self) -> str:
        captures_str = f", captured {len(self.captured_positions)} stones" if self.has_captures() else ""
        return f"MoveDelta(player {self.player} at {self.position}{captures_str})"