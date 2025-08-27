"""
Fast move generation with distance ordering for high-performance MCTS.
"""

from typing import List, Tuple, Optional
from .game_state import GameState
from .distance_rings import get_distance_rings

class MoveGenerator:
    """
    High-performance move generator with progressive widening support.
    
    Key features:
    - Distance-based ordering (closer to existing stones first)
    - Progressive widening based on node visit counts
    - O(1) distance lookups using precomputed rings
    - Excludes occupied positions automatically
    """
    
    def __init__(self, board_size: int = 19):
        self.board_size = board_size
        self.distance_rings = get_distance_rings(board_size)
    
    def get_ordered_moves(self, game_state: GameState, max_distance: int = 3) -> List[Tuple[int, int]]:
        """
        Get legal moves ordered by distance to existing stones.
        
        This is the core method for MCTS move generation:
        - Returns moves closer to stones first (more likely to be good)
        - Automatically excludes occupied positions
        - Respects tournament rule if enabled
        
        Args:
            game_state: Current game state
            max_distance: Maximum distance from existing stones
            
        Returns:
            List of (row, col) positions ordered by distance
        """
        # Get all stone positions from move history
        stone_positions = []
        for delta in game_state.move_history:
            stone_positions.append(delta.position)
        
        if not stone_positions:
            # No stones yet - return all legal moves (handles tournament rule)
            return game_state.get_legal_moves()
        
        # Get positions ordered by distance to existing stones
        ordered_candidates = self.distance_rings.get_ordered_positions_around_stones(
            stone_positions, max_distance
        )
        
        # Filter to only legal moves (empty positions + tournament rule)
        legal_moves = set(game_state.get_legal_moves())
        ordered_legal_moves = [pos for pos in ordered_candidates if pos in legal_moves]
        
        return ordered_legal_moves
    
    def get_progressive_widening_moves(self, game_state: GameState, node_visits: int) -> List[Tuple[int, int]]:
        """
        Get moves using progressive widening strategy.
        
        Progressive widening thresholds (from spec):
        - <10 visits: distance 1, ~15 moves
        - <100 visits: distance 2, ~30 moves  
        - <1000 visits: distance 3, ~50 moves
        - 1000+ visits: distance 5, ~80 moves
        
        Args:
            game_state: Current game state
            node_visits: Number of times this node has been visited
            
        Returns:
            List of moves appropriate for the visit count
        """
        if node_visits < 10:
            # Early exploration: very close moves only
            max_distance = 1
            target_moves = 15
        elif node_visits < 100:
            # Medium exploration: nearby moves
            max_distance = 2  
            target_moves = 30
        elif node_visits < 1000:
            # Deep exploration: wider area
            max_distance = 3
            target_moves = 50
        else:
            # Full exploration: consider distant moves
            max_distance = 5
            target_moves = 80
        
        ordered_moves = self.get_ordered_moves(game_state, max_distance)
        
        # Limit to target number of moves (take the closest ones)
        if len(ordered_moves) > target_moves:
            ordered_moves = ordered_moves[:target_moves]
        
        return ordered_moves
    
    def get_immediate_moves(self, game_state: GameState) -> List[Tuple[int, int]]:
        """
        Get moves for immediate threats/opportunities (distance 1 only).
        
        Useful for tactical analysis and quick evaluation.
        
        Args:
            game_state: Current game state
            
        Returns:
            List of moves at distance 1 from existing stones
        """
        return self.get_ordered_moves(game_state, max_distance=1)
    
    def get_all_legal_moves_ordered(self, game_state: GameState) -> List[Tuple[int, int]]:
        """
        Get ALL legal moves ordered by distance to stones.
        
        For final search phases or when we want to consider all possibilities.
        Uses a large max_distance to include the entire board.
        
        Args:
            game_state: Current game state
            
        Returns:
            All legal moves ordered by distance to existing stones
        """
        return self.get_ordered_moves(game_state, max_distance=self.board_size)
    
    def estimate_move_count_at_distance(self, game_state: GameState, distance: int) -> int:
        """
        Estimate how many moves would be available at a specific distance.
        
        Useful for tuning progressive widening parameters.
        
        Args:
            game_state: Current game state  
            distance: Distance to query
            
        Returns:
            Estimated number of moves at that distance
        """
        stone_positions = [delta.position for delta in game_state.move_history]
        
        if not stone_positions:
            return 0
        
        # Get positions at exactly this distance
        positions_at_distance = []
        for stone_pos in stone_positions:
            row, col = stone_pos
            distance_positions = self.distance_rings.get_positions_at_distance(row, col, distance)
            positions_at_distance.extend(distance_positions)
        
        # Remove duplicates and occupied positions
        unique_positions = set(positions_at_distance)
        legal_moves = set(game_state.get_legal_moves())
        available_at_distance = unique_positions.intersection(legal_moves)
        
        return len(available_at_distance)
    
    def get_move_statistics(self, game_state: GameState) -> dict:
        """
        Get statistics about move generation for debugging/tuning.
        
        Args:
            game_state: Current game state
            
        Returns:
            Dictionary with move generation statistics
        """
        total_legal = len(game_state.get_legal_moves())
        stone_count = len(game_state.move_history)
        
        stats = {
            'total_legal_moves': total_legal,
            'stone_count': stone_count,
            'moves_by_distance': {}
        }
        
        # Count moves at each distance
        for distance in range(1, min(6, self.board_size)):  # Check first 5 distances
            count = self.estimate_move_count_at_distance(game_state, distance)
            stats['moves_by_distance'][distance] = count
        
        # Progressive widening estimates
        stats['progressive_widening'] = {
            'early_exploration': len(self.get_progressive_widening_moves(game_state, 5)),
            'medium_exploration': len(self.get_progressive_widening_moves(game_state, 50)), 
            'deep_exploration': len(self.get_progressive_widening_moves(game_state, 500)),
            'full_exploration': len(self.get_progressive_widening_moves(game_state, 5000))
        }
        
        return stats

# Global instance for efficiency
_global_move_generator: Optional[MoveGenerator] = None

def get_move_generator(board_size: int = 19) -> MoveGenerator:
    """
    Get global move generator instance.
    Creates it once at startup for efficiency.
    """
    global _global_move_generator
    if _global_move_generator is None or _global_move_generator.board_size != board_size:
        _global_move_generator = MoveGenerator(board_size)
    return _global_move_generator