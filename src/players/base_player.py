from abc import ABC, abstractmethod
from typing import Tuple, Optional
from games.pente import Pente

class BasePlayer(ABC):
    """Abstract base class for all player types."""
    
    def __init__(self, name: str, player_id: int):
        """
        Initialize player.
        
        Args:
            name: Human-readable name for the player
            player_id: Player identifier (1 or -1)
        """
        self.name = name
        self.player_id = player_id
        
    @abstractmethod
    def get_move(self, game: Pente) -> Tuple[int, int]:
        """
        Get the next move for this player.
        
        Args:
            game: Current game state
            
        Returns:
            Tuple of (row, col) coordinates for the move
        """
        pass
    
    @abstractmethod
    def is_human(self) -> bool:
        """
        Check if this player requires human input.
        
        Returns:
            True if player is human, False if AI
        """
        pass
    
    def reset(self) -> None:
        """
        Reset player state for a new game.
        Override if player needs to clear internal state.
        """
        pass
    
    def get_name(self) -> str:
        """Get player name."""
        return self.name
    
    def get_player_id(self) -> int:
        """Get player ID (1 or -1)."""
        return self.player_id
    
    def notify_move(self, move: Tuple[int, int], player_id: int) -> None:
        """
        Notify player of a move made in the game.
        Override if player needs to track game history.
        
        Args:
            move: The move that was made
            player_id: ID of player who made the move
        """
        pass
    
    def notify_game_end(self, winner: Optional[int]) -> None:
        """
        Notify player that the game has ended.
        Override if player needs to handle game completion.
        
        Args:
            winner: Player ID of winner, or None for draw
        """
        pass