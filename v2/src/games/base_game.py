from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import numpy as np

class BaseGame(ABC):
    """Abstract base class for board games."""
    
    def __init__(self, board_size: int):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=int)
        self.current_player = 1
        self.captures = {1: 0, -1: 0}
        self.move_history = []
        
    @abstractmethod
    def get_legal_moves(self) -> List[Tuple[int, int]]:
        """Return list of legal moves."""
        pass
    
    @abstractmethod
    def make_move(self, move: Tuple[int, int]) -> None:
        """Execute a move on the board."""
        pass
    
    @abstractmethod
    def undo_move(self) -> None:
        """Undo the last move."""
        pass
    
    @abstractmethod
    def is_terminal(self) -> bool:
        """Check if game has ended."""
        pass
    
    @abstractmethod
    def get_winner(self) -> Optional[int]:
        """Return winner (1, -1) or None."""
        pass
    
    @abstractmethod
    def clone(self) -> 'BaseGame':
        """Create a deep copy of the game state."""
        pass
    
    def get_board_hash(self) -> int:
        """Get hash of current board state for transposition table."""
        return hash(self.board.tobytes())