"""
Precomputed distance rings for efficient move generation.
Critical for high-performance MCTS implementation.
"""

from typing import List, Tuple, Dict
import time

class DistanceRings:
    """
    Precomputed distance rings for 19x19 board.
    
    Instead of calculating distances during MCTS (O(n) per query),
    we precompute all distances once at startup (~50ms) and get O(1) lookups.
    
    This enables efficient progressive widening based on distance from existing stones.
    """
    
    def __init__(self, board_size: int = 19):
        self.board_size = board_size
        self.max_distance = board_size - 1  # Maximum possible distance on board
        
        # rings[position][distance] = list of positions at exactly that distance
        # Position is encoded as row * board_size + col for efficiency
        self.rings: Dict[int, Dict[int, List[Tuple[int, int]]]] = {}
        
        print(f"Precomputing distance rings for {board_size}x{board_size} board...")
        start_time = time.time()
        self._precompute_all_distances()
        end_time = time.time()
        print(f"Distance rings computed in {(end_time - start_time) * 1000:.1f}ms")
    
    def _precompute_all_distances(self):
        """Precompute distance rings for all positions."""
        for row in range(self.board_size):
            for col in range(self.board_size):
                position_id = self._encode_position(row, col)
                self.rings[position_id] = {}
                
                # Initialize distance buckets
                for distance in range(1, self.max_distance + 1):
                    self.rings[position_id][distance] = []
                
                # Calculate distances to all other positions
                for other_row in range(self.board_size):
                    for other_col in range(self.board_size):
                        if other_row == row and other_col == col:
                            continue  # Skip same position
                        
                        # Use Chebyshev distance (max of row/col differences)
                        # This is appropriate for Pente as stones can affect diagonally
                        distance = max(abs(other_row - row), abs(other_col - col))
                        
                        if distance <= self.max_distance:
                            self.rings[position_id][distance].append((other_row, other_col))
    
    def _encode_position(self, row: int, col: int) -> int:
        """Encode position as single integer for efficient storage."""
        return row * self.board_size + col
    
    def get_positions_at_distance(self, center_row: int, center_col: int, distance: int) -> List[Tuple[int, int]]:
        """
        Get all positions at exactly the specified distance from center.
        
        Args:
            center_row: Center row position
            center_col: Center col position  
            distance: Exact distance to query
            
        Returns:
            List of (row, col) positions at that distance
        """
        if not (0 <= center_row < self.board_size and 0 <= center_col < self.board_size):
            return []
        
        if distance < 1 or distance > self.max_distance:
            return []
        
        position_id = self._encode_position(center_row, center_col)
        return self.rings[position_id][distance]
    
    def get_positions_within_distance(self, center_row: int, center_col: int, max_distance: int) -> List[Tuple[int, int]]:
        """
        Get all positions within the specified maximum distance from center.
        
        Args:
            center_row: Center row position
            center_col: Center col position
            max_distance: Maximum distance (inclusive)
            
        Returns:
            List of (row, col) positions within that distance, ordered by distance
        """
        if not (0 <= center_row < self.board_size and 0 <= center_col < self.board_size):
            return []
        
        if max_distance < 1:
            return []
        
        positions = []
        position_id = self._encode_position(center_row, center_col)
        
        for distance in range(1, min(max_distance + 1, self.max_distance + 1)):
            positions.extend(self.rings[position_id][distance])
        
        return positions
    
    def get_distance(self, row1: int, col1: int, row2: int, col2: int) -> int:
        """
        Get Chebyshev distance between two positions.
        
        This is a simple calculation but provided for completeness.
        For performance-critical code, consider caching if needed.
        """
        return max(abs(row2 - row1), abs(col2 - col1))
    
    def get_ordered_positions_around_stones(self, stone_positions: List[Tuple[int, int]], 
                                           max_distance: int = 3) -> List[Tuple[int, int]]:
        """
        Get positions ordered by minimum distance to any existing stone.
        
        This is the key method for efficient move generation in MCTS:
        - Positions closer to existing stones are returned first
        - Duplicates are removed 
        - Critical for progressive widening
        
        Args:
            stone_positions: List of (row, col) positions with existing stones
            max_distance: Maximum distance to consider
            
        Returns:
            List of positions ordered by distance to nearest stone
        """
        if not stone_positions:
            return []
        
        # Track positions by their minimum distance to any stone
        distance_buckets: Dict[int, List[Tuple[int, int]]] = {}
        seen_positions = set()
        stone_positions_set = set(stone_positions)  # For quick lookup
        
        # Initialize distance buckets
        for distance in range(1, max_distance + 1):
            distance_buckets[distance] = []
        
        # For each existing stone, add positions at each distance
        for stone_row, stone_col in stone_positions:
            if not (0 <= stone_row < self.board_size and 0 <= stone_col < self.board_size):
                continue
                
            position_id = self._encode_position(stone_row, stone_col)
            
            for distance in range(1, max_distance + 1):
                if distance > self.max_distance:
                    break
                    
                for pos in self.rings[position_id][distance]:
                    if pos not in seen_positions and pos not in stone_positions_set:
                        # This position hasn't been seen at a closer distance and isn't a stone
                        distance_buckets[distance].append(pos)
                        seen_positions.add(pos)
        
        # Combine buckets in distance order
        ordered_positions = []
        for distance in range(1, max_distance + 1):
            ordered_positions.extend(distance_buckets[distance])
        
        return ordered_positions
    
    def get_statistics(self) -> Dict[str, int]:
        """Get statistics about the precomputed rings."""
        total_positions = self.board_size * self.board_size
        total_ring_entries = 0
        
        for position_id in self.rings:
            for distance in self.rings[position_id]:
                total_ring_entries += len(self.rings[position_id][distance])
        
        return {
            'board_size': self.board_size,
            'total_positions': total_positions,
            'max_distance': self.max_distance,
            'total_ring_entries': total_ring_entries,
            'memory_positions': len(self.rings)
        }
    
    def __str__(self) -> str:
        """String representation with statistics."""
        stats = self.get_statistics()
        return (f"DistanceRings({stats['board_size']}x{stats['board_size']}, "
                f"{stats['total_ring_entries']} precomputed distances)")

# Global instance for efficiency - created once at startup
_global_distance_rings: DistanceRings = None

def get_distance_rings(board_size: int = 19) -> DistanceRings:
    """
    Get global distance rings instance.
    Creates it once at startup for efficiency.
    """
    global _global_distance_rings
    if _global_distance_rings is None or _global_distance_rings.board_size != board_size:
        _global_distance_rings = DistanceRings(board_size)
    return _global_distance_rings