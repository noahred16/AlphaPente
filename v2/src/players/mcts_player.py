from typing import Tuple
from .base_player import BasePlayer
from games.pente import Pente
from mcts import MCTS

class MCTSPlayer(BasePlayer):
    """AI player using Monte Carlo Tree Search."""
    
    def __init__(self, name: str = "MCTS AI", player_id: int = -1, 
                 max_iterations: int = 1000, exploration_weight: float = 1.414):
        """
        Initialize MCTS player.
        
        Args:
            name: Player name
            player_id: Player identifier (1 or -1)
            max_iterations: Number of MCTS iterations per move
            exploration_weight: UCB1 exploration parameter
        """
        super().__init__(name, player_id)
        self.max_iterations = max_iterations
        self.exploration_weight = exploration_weight
        self.mcts = MCTS(exploration_weight=exploration_weight, 
                        max_iterations=max_iterations)
    
    def get_move(self, game: Pente) -> Tuple[int, int]:
        """
        Get the best move using MCTS.
        
        Args:
            game: Current game state
            
        Returns:
            Best move coordinates (row, col)
        """
        # Ensure the game state reflects this player's turn
        if game.current_player != self.player_id:
            raise ValueError(f"Not {self.name}'s turn (current: {game.current_player}, expected: {self.player_id})")
        
        # Run MCTS to find best move
        move = self.mcts.search(game)
        return move
    
    def is_human(self) -> bool:
        """MCTS player is AI."""
        return False
    
    def set_thinking_time(self, max_iterations: int) -> None:
        """
        Adjust the thinking time by changing max iterations.
        
        Args:
            max_iterations: New iteration limit
        """
        self.max_iterations = max_iterations
        self.mcts.max_iterations = max_iterations
    
    def set_exploration_weight(self, exploration_weight: float) -> None:
        """
        Adjust exploration vs exploitation balance.
        
        Args:
            exploration_weight: New UCB1 exploration parameter
        """
        self.exploration_weight = exploration_weight
        self.mcts.exploration_weight = exploration_weight
    
    def get_move_statistics(self, game: Pente) -> list:
        """
        Get detailed statistics for all explored moves.
        
        Args:
            game: Current game state
            
        Returns:
            List of (move, visits, win_rate) tuples
        """
        return self.mcts.get_move_statistics(game)

class EasyMCTSPlayer(MCTSPlayer):
    """Easy difficulty MCTS player."""
    
    def __init__(self, player_id: int = -1):
        super().__init__(name="Easy AI", player_id=player_id, 
                        max_iterations=100, exploration_weight=1.0)

class MediumMCTSPlayer(MCTSPlayer):
    """Medium difficulty MCTS player."""
    
    def __init__(self, player_id: int = -1):
        super().__init__(name="Medium AI", player_id=player_id, 
                        max_iterations=500, exploration_weight=1.414)

class HardMCTSPlayer(MCTSPlayer):
    """Hard difficulty MCTS player."""
    
    def __init__(self, player_id: int = -1):
        super().__init__(name="Hard AI", player_id=player_id, 
                        max_iterations=2000, exploration_weight=1.414)

class ExpertMCTSPlayer(MCTSPlayer):
    """Expert difficulty MCTS player."""
    
    def __init__(self, player_id: int = -1):
        super().__init__(name="Expert AI", player_id=player_id, 
                        max_iterations=5000, exploration_weight=1.2)