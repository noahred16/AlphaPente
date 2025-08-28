"""
Tests for DistanceRings precomputation system.
"""

import pytest
from src.core.distance_rings import DistanceRings, get_distance_rings

class TestDistanceRings:
    """Test suite for DistanceRings functionality."""
    
    def test_initialization_small_board(self):
        """Test DistanceRings initialization with small board."""
        rings = DistanceRings(board_size=5)
        
        assert rings.board_size == 5
        assert rings.max_distance == 4  # 5-1
        assert len(rings.rings) == 25   # 5x5 positions
        
        # Check that each position has distance buckets
        for row in range(5):
            for col in range(5):
                position_id = rings._encode_position(row, col)
                assert position_id in rings.rings
                assert len(rings.rings[position_id]) == 4  # distances 1-4
    
    def test_encode_position(self):
        """Test position encoding."""
        rings = DistanceRings(5)
        
        assert rings._encode_position(0, 0) == 0
        assert rings._encode_position(0, 4) == 4
        assert rings._encode_position(4, 0) == 20
        assert rings._encode_position(4, 4) == 24
        assert rings._encode_position(2, 3) == 13  # 2*5 + 3
    
    def test_get_distance_calculation(self):
        """Test distance calculation (Chebyshev distance)."""
        rings = DistanceRings(5)
        
        # Same position
        assert rings.get_distance(2, 2, 2, 2) == 0
        
        # Adjacent positions (distance 1)
        assert rings.get_distance(2, 2, 2, 3) == 1  # Horizontal
        assert rings.get_distance(2, 2, 3, 2) == 1  # Vertical
        assert rings.get_distance(2, 2, 3, 3) == 1  # Diagonal
        
        # Distance 2
        assert rings.get_distance(2, 2, 2, 4) == 2  # Horizontal
        assert rings.get_distance(2, 2, 4, 2) == 2  # Vertical
        assert rings.get_distance(2, 2, 4, 4) == 2  # Diagonal
        
        # Mixed distances - should use max
        assert rings.get_distance(0, 0, 2, 1) == 2  # max(2, 1) = 2
        assert rings.get_distance(0, 0, 1, 3) == 3  # max(1, 3) = 3
    
    def test_get_positions_at_distance_center(self):
        """Test getting positions at specific distance from center."""
        rings = DistanceRings(5)
        center = (2, 2)  # Center of 5x5 board
        
        # Distance 1 from center should have 8 positions
        dist1_positions = rings.get_positions_at_distance(2, 2, 1)
        expected_dist1 = [
            (1, 1), (1, 2), (1, 3),
            (2, 1),         (2, 3),
            (3, 1), (3, 2), (3, 3)
        ]
        
        assert len(dist1_positions) == 8
        for pos in expected_dist1:
            assert pos in dist1_positions
    
    def test_get_positions_at_distance_corner(self):
        """Test getting positions at distance from corner."""
        rings = DistanceRings(5)
        
        # Distance 1 from corner (0,0)
        dist1_positions = rings.get_positions_at_distance(0, 0, 1)
        expected_dist1 = [(0, 1), (1, 0), (1, 1)]
        
        assert len(dist1_positions) == 3
        for pos in expected_dist1:
            assert pos in dist1_positions
    
    def test_get_positions_at_distance_invalid(self):
        """Test invalid inputs to get_positions_at_distance."""
        rings = DistanceRings(5)
        
        # Invalid position
        assert rings.get_positions_at_distance(-1, 0, 1) == []
        assert rings.get_positions_at_distance(5, 0, 1) == []
        assert rings.get_positions_at_distance(0, -1, 1) == []
        assert rings.get_positions_at_distance(0, 5, 1) == []
        
        # Invalid distance
        assert rings.get_positions_at_distance(2, 2, 0) == []
        assert rings.get_positions_at_distance(2, 2, -1) == []
        assert rings.get_positions_at_distance(2, 2, 10) == []  # Beyond max distance
    
    def test_get_positions_within_distance(self):
        """Test getting all positions within max distance."""
        rings = DistanceRings(5)
        
        # Within distance 1 from center
        positions = rings.get_positions_within_distance(2, 2, 1)
        assert len(positions) == 8  # All distance-1 neighbors
        
        # Within distance 2 from corner
        positions = rings.get_positions_within_distance(0, 0, 2)
        
        # Should include distance-1 and distance-2 positions
        # Distance 1: (0,1), (1,0), (1,1) = 3 positions
        # Distance 2: (0,2), (2,0), (1,2), (2,1), (2,2) = 5 positions
        # Total: 8 positions
        assert len(positions) == 8
        
        # Verify ordering (distance 1 first, then distance 2)
        distance_1_positions = rings.get_positions_at_distance(0, 0, 1)
        for i, pos in enumerate(positions[:3]):  # First 3 should be distance 1
            assert pos in distance_1_positions
    
    def test_get_ordered_positions_around_single_stone(self):
        """Test ordered positions around a single stone."""
        rings = DistanceRings(5)
        
        stone_positions = [(2, 2)]  # Center stone
        ordered_positions = rings.get_ordered_positions_around_stones(stone_positions, max_distance=2)
        
        # Should have distance-1 positions first, then distance-2
        dist1_positions = rings.get_positions_at_distance(2, 2, 1)
        dist2_positions = rings.get_positions_at_distance(2, 2, 2)
        
        expected_total = len(dist1_positions) + len(dist2_positions)
        assert len(ordered_positions) == expected_total
        
        # Verify ordering: first 8 should be distance 1
        for pos in ordered_positions[:8]:
            assert pos in dist1_positions
    
    def test_get_ordered_positions_around_multiple_stones(self):
        """Test ordered positions around multiple stones."""
        rings = DistanceRings(7)  # Slightly larger board
        
        stone_positions = [(2, 2), (4, 4)]  # Two stones
        ordered_positions = rings.get_ordered_positions_around_stones(stone_positions, max_distance=2)
        
        # Should have no duplicates
        assert len(ordered_positions) == len(set(ordered_positions))
        
        # All positions should be within distance 2 of at least one stone
        for pos in ordered_positions:
            row, col = pos
            min_distance = min(
                rings.get_distance(row, col, 2, 2),
                rings.get_distance(row, col, 4, 4)
            )
            assert min_distance <= 2
            assert min_distance >= 1  # Should not include stone positions themselves
    
    def test_get_ordered_positions_empty_stones(self):
        """Test ordered positions with no stones."""
        rings = DistanceRings(5)
        
        ordered_positions = rings.get_ordered_positions_around_stones([], max_distance=3)
        assert ordered_positions == []
    
    def test_get_ordered_positions_invalid_stones(self):
        """Test ordered positions with invalid stone positions."""
        rings = DistanceRings(5)
        
        # Mix of valid and invalid positions
        stone_positions = [(-1, 0), (2, 2), (10, 10), (3, 3)]
        ordered_positions = rings.get_ordered_positions_around_stones(stone_positions, max_distance=1)
        
        # Should only process valid stone positions (2,2) and (3,3)
        assert len(ordered_positions) > 0
        
        # All returned positions should be valid
        for pos in ordered_positions:
            row, col = pos
            assert 0 <= row < 5
            assert 0 <= col < 5
    
    def test_progressive_widening_simulation(self):
        """Test simulation of progressive widening strategy."""
        rings = DistanceRings(19)
        
        # Simulate game with a few stones
        stone_positions = [(9, 9), (10, 10), (8, 8)]  # Center area
        
        # Progressive widening thresholds from spec:
        # <10 visits: distance 1, ~15 moves
        # <100 visits: distance 2, ~30 moves
        # <1000 visits: distance 3, ~50 moves
        
        dist1_moves = rings.get_ordered_positions_around_stones(stone_positions, max_distance=1)
        dist2_moves = rings.get_ordered_positions_around_stones(stone_positions, max_distance=2)
        dist3_moves = rings.get_ordered_positions_around_stones(stone_positions, max_distance=3)
        
        # Should have increasing numbers of moves
        assert len(dist1_moves) < len(dist2_moves) < len(dist3_moves)
        
        # Distance 1 should be reasonable size for initial exploration
        assert 10 <= len(dist1_moves) <= 30  # Around 15 target
        
        print(f"Progressive widening moves: d1={len(dist1_moves)}, d2={len(dist2_moves)}, d3={len(dist3_moves)}")
    
    def test_get_statistics(self):
        """Test statistics generation."""
        rings = DistanceRings(5)
        stats = rings.get_statistics()
        
        assert stats['board_size'] == 5
        assert stats['total_positions'] == 25
        assert stats['max_distance'] == 4
        assert stats['memory_positions'] == 25
        assert stats['total_ring_entries'] > 0
    
    def test_string_representation(self):
        """Test string representation."""
        rings = DistanceRings(5)
        str_repr = str(rings)
        
        assert '5x5' in str_repr
        assert 'DistanceRings' in str_repr
        assert 'precomputed distances' in str_repr
    
    def test_global_instance(self):
        """Test global distance rings instance."""
        # First call should create instance
        rings1 = get_distance_rings(5)
        assert isinstance(rings1, DistanceRings)
        assert rings1.board_size == 5
        
        # Second call should return same instance
        rings2 = get_distance_rings(5)
        assert rings2 is rings1
        
        # Different size should create new instance
        rings3 = get_distance_rings(7)
        assert rings3 is not rings1
        assert rings3.board_size == 7
    
    def test_large_board_performance(self):
        """Test that large board initialization completes reasonably quickly."""
        import time
        
        start_time = time.time()
        rings = DistanceRings(19)  # Full Pente board
        end_time = time.time()
        
        # Should complete within reasonable time (spec says ~50ms target)
        elapsed_ms = (end_time - start_time) * 1000
        print(f"19x19 board initialized in {elapsed_ms:.1f}ms")
        
        # Generous timeout for CI/test environments
        assert elapsed_ms < 1000  # Should be much faster, but allow for slow CI
        
        # Verify it actually computed everything
        assert len(rings.rings) == 361  # 19x19 positions
        
        # Test a few lookups work correctly
        center_positions = rings.get_positions_at_distance(9, 9, 1)
        assert len(center_positions) == 8  # 8 neighbors for center position
    
    def test_distance_accuracy(self):
        """Test that precomputed distances match calculated distances."""
        rings = DistanceRings(7)
        
        # Test several random positions
        test_positions = [(0, 0), (3, 3), (2, 5), (6, 1)]
        
        for center_row, center_col in test_positions:
            for distance in range(1, rings.max_distance + 1):
                precomputed_positions = rings.get_positions_at_distance(center_row, center_col, distance)
                
                # Verify each precomputed position is actually at the claimed distance
                for pos_row, pos_col in precomputed_positions:
                    calculated_distance = rings.get_distance(center_row, center_col, pos_row, pos_col)
                    assert calculated_distance == distance, f"Position {pos_row},{pos_col} from {center_row},{center_col}: expected {distance}, got {calculated_distance}"