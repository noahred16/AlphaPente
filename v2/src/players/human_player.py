import threading
from typing import Tuple, Optional
from .base_player import BasePlayer
from games.pente import Pente

class HumanPlayer(BasePlayer):
    """Human player that receives moves through web interface."""
    
    def __init__(self, name: str = "Human", player_id: int = 1):
        super().__init__(name, player_id)
        self._pending_move: Optional[Tuple[int, int]] = None
        self._move_event = threading.Event()
        self._waiting_for_move = False
    
    def get_move(self, game: Pente) -> Tuple[int, int]:
        """
        Wait for a move to be submitted via the web interface.
        
        Args:
            game: Current game state
            
        Returns:
            The move submitted by the human player
        """
        # Reset event and wait for move
        self._move_event.clear()
        self._waiting_for_move = True
        self._pending_move = None
        
        # This will block until submit_move() is called
        self._move_event.wait()
        
        self._waiting_for_move = False
        move = self._pending_move
        self._pending_move = None
        
        if move is None:
            raise ValueError("No move was submitted")
            
        return move
    
    def submit_move(self, move: Tuple[int, int]) -> bool:
        """
        Submit a move from the web interface.
        
        Args:
            move: The move coordinates (row, col)
            
        Returns:
            True if move was accepted, False if not waiting for move
        """
        if not self._waiting_for_move:
            return False
            
        self._pending_move = move
        self._move_event.set()
        return True
    
    def is_waiting_for_move(self) -> bool:
        """Check if player is currently waiting for a move."""
        return self._waiting_for_move
    
    def cancel_move_request(self) -> None:
        """Cancel any pending move request (e.g., for game reset)."""
        if self._waiting_for_move:
            self._waiting_for_move = False
            self._move_event.set()
    
    def is_human(self) -> bool:
        """Human player returns True."""
        return True
    
    def reset(self) -> None:
        """Reset for new game."""
        self.cancel_move_request()
        self._pending_move = None