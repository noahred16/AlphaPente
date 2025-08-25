"""
Rollout policy for MCTS simulations.
Implements random rollouts with early termination for performance.
"""

import random
from typing import List, Tuple, Optional
from ..core.game_state import GameState

class RolloutPolicy:
    """
    High-performance rollout policy for MCTS simulations.
    
    Key features:
    - Random move selection from legal moves
    - Early termination when game is decided
    - Optimized for speed (critical path in MCTS)
    - Uses single GameState with make/undo (no copying!)
    """
    
    def __init__(self, max_moves: int = 200, seed: Optional[int] = None):
        """
        Initialize rollout policy.
        
        Args:
            max_moves: Maximum moves in a rollout (prevents infinite games)
            seed: Random seed for reproducibility (None for random)
        """
        self.max_moves = max_moves
        
        if seed is not None:
            random.seed(seed)
    
    def rollout(self, game_state: GameState) -> float:
        """
        Perform a random rollout from the current game state.
        
        CRITICAL: This modifies the game state! Caller must undo all moves.
        
        Args:
            game_state: Current game state (will be modified!)
            
        Returns:
            Rollout result from current player's perspective:
            - 1.0: Current player wins
            - 0.0: Current player loses  
            - 0.5: Draw/timeout
        """
        moves_made = []
        rollout_player = game_state.current_player
        
        try:
            for _ in range(self.max_moves):
                # Check if game is already over
                if game_state.is_terminal():
                    winner = game_state.get_winner()
                    if winner is None:
                        result = 0.5  # Draw
                    elif winner == rollout_player:
                        result = 1.0  # Win for rollout player
                    else:
                        result = 0.0  # Loss for rollout player
                    break
                
                # Get legal moves
                legal_moves = game_state.get_legal_moves()
                if not legal_moves:
                    result = 0.5  # Draw - no moves available
                    break
                
                # Select random move
                move = random.choice(legal_moves)
                
                # Make the move
                delta = game_state.make_move(move)
                moves_made.append(delta)
                
                # Early termination check - if current player just won
                if game_state.is_terminal():
                    winner = game_state.get_winner()
                    if winner is None:
                        result = 0.5  # Draw
                    elif winner == rollout_player:
                        result = 1.0  # Win for rollout player
                    else:
                        result = 0.0  # Loss for rollout player
                    break
            else:
                # Rollout reached max moves without termination
                result = 0.5  # Treat as draw
        
        finally:
            # CRITICAL: Undo all moves to restore original state
            for _ in range(len(moves_made)):
                game_state.undo_move()
        
        return result
    
    def rollout_with_bias(self, game_state: GameState, bias_moves: List[Tuple[int, int]] = None,
                         bias_probability: float = 0.3) -> float:
        """
        Perform rollout with bias toward specific moves (for advanced play).
        
        Args:
            game_state: Current game state (will be modified!)
            bias_moves: Moves to bias toward (e.g., captures, threats)
            bias_probability: Probability of selecting biased move when available
            
        Returns:
            Rollout result from current player's perspective
        """
        if bias_moves is None:
            bias_moves = []
        
        moves_made = []
        rollout_player = game_state.current_player
        
        try:
            for _ in range(self.max_moves):
                if game_state.is_terminal():
                    winner = game_state.get_winner()
                    if winner is None:
                        result = 0.5
                    elif winner == rollout_player:
                        result = 1.0
                    else:
                        result = 0.0
                    break
                
                legal_moves = game_state.get_legal_moves()
                if not legal_moves:
                    result = 0.5
                    break
                
                # Check for biased moves
                available_bias_moves = [m for m in bias_moves if m in legal_moves]
                
                if available_bias_moves and random.random() < bias_probability:
                    # Select biased move
                    move = random.choice(available_bias_moves)
                else:
                    # Select random move
                    move = random.choice(legal_moves)
                
                delta = game_state.make_move(move)
                moves_made.append(delta)
                
                if game_state.is_terminal():
                    winner = game_state.get_winner()
                    if winner is None:
                        result = 0.5
                    elif winner == rollout_player:
                        result = 1.0
                    else:
                        result = 0.0
                    break
            else:
                result = 0.5
        
        finally:
            # Undo all moves
            for _ in range(len(moves_made)):
                game_state.undo_move()
        
        return result
    
    def quick_evaluation(self, game_state: GameState) -> Optional[float]:
        """
        Quick evaluation without full rollout (for early termination).
        
        Args:
            game_state: Current game state
            
        Returns:
            Quick evaluation result, or None if full rollout needed:
            - 1.0: Clear win for current player
            - 0.0: Clear loss for current player
            - None: Needs full rollout
        """
        if game_state.is_terminal():
            winner = game_state.get_winner()
            current_player = game_state.current_player
            
            if winner is None:
                return 0.5  # Draw
            elif winner == current_player:
                return 1.0  # Win
            else:
                return 0.0  # Loss
        
        # Check for immediate wins/losses
        # This could be expanded with threat detection, capture analysis, etc.
        
        # For now, just check capture counts
        current_player = game_state.current_player
        opponent = -current_player
        
        current_captures = game_state.captures[current_player]
        opponent_captures = game_state.captures[opponent]
        
        # If someone is about to win by captures
        if current_captures >= game_state.captures_to_win - 1:
            # Current player might win quickly
            return None  # Need rollout to confirm
        elif opponent_captures >= game_state.captures_to_win - 1:
            # Opponent might win quickly  
            return None  # Need rollout to confirm
        
        return None  # Need full rollout
    
    def get_statistics(self) -> dict:
        """Get rollout policy statistics."""
        return {
            'max_moves': self.max_moves,
            'policy_type': 'random_rollout'
        }
    
    def __str__(self) -> str:
        """String representation."""
        return f"RolloutPolicy(max_moves={self.max_moves})"