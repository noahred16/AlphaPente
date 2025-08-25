"""
Tests for MoveDelta class.
"""

import pytest
from src.core.move_delta import MoveDelta

class TestMoveDelta:
    """Test suite for MoveDelta functionality."""
    
    def test_basic_initialization(self):
        """Test basic MoveDelta initialization."""
        delta = MoveDelta((5, 10), 1)
        
        assert delta.position == (5, 10)
        assert delta.player == 1
        assert delta.captured_positions == []
        assert delta.captured_player == 0
        assert delta.capture_count == 0
        assert not delta.has_captures()
    
    def test_initialization_with_captures(self):
        """Test MoveDelta initialization with captures."""
        captured_pos = [(3, 4), (3, 5)]
        delta = MoveDelta(
            position=(3, 6),
            player=1,
            captured_positions=captured_pos,
            captured_player=-1,
            capture_count=1
        )
        
        assert delta.position == (3, 6)
        assert delta.player == 1
        assert delta.captured_positions == captured_pos
        assert delta.captured_player == -1
        assert delta.capture_count == 1
        assert delta.has_captures()
    
    def test_initialization_with_keyword_args(self):
        """Test initialization using keyword arguments."""
        delta = MoveDelta(
            position=(10, 10),
            player=-1,
            captured_positions=[(8, 8), (9, 9)],
            captured_player=1,
            capture_count=1
        )
        
        assert delta.position == (10, 10)
        assert delta.player == -1
        assert len(delta.captured_positions) == 2
        assert delta.captured_player == 1
        assert delta.capture_count == 1
    
    def test_has_captures_empty(self):
        """Test has_captures with no captures."""
        delta = MoveDelta((0, 0), 1)
        assert not delta.has_captures()
        
        # Even with empty list explicitly set
        delta = MoveDelta((0, 0), 1, captured_positions=[])
        assert not delta.has_captures()
    
    def test_has_captures_with_captures(self):
        """Test has_captures with actual captures."""
        delta = MoveDelta((0, 0), 1, captured_positions=[(1, 1)])
        assert delta.has_captures()
        
        delta = MoveDelta((0, 0), 1, captured_positions=[(1, 1), (2, 2), (3, 3)])
        assert delta.has_captures()
    
    def test_repr_no_captures(self):
        """Test string representation without captures."""
        delta = MoveDelta((7, 8), -1)
        repr_str = repr(delta)
        
        assert "player -1" in repr_str
        assert "(7, 8)" in repr_str
        assert "captured" not in repr_str
    
    def test_repr_with_captures(self):
        """Test string representation with captures."""
        delta = MoveDelta(
            (10, 15),
            1,
            captured_positions=[(8, 9), (8, 10)],
            captured_player=-1,
            capture_count=1
        )
        repr_str = repr(delta)
        
        assert "player 1" in repr_str
        assert "(10, 15)" in repr_str
        assert "captured 2 stones" in repr_str
    
    def test_default_values(self):
        """Test that default values work correctly."""
        delta = MoveDelta((0, 0), 1)
        
        # Test all defaults
        assert delta.captured_positions == []
        assert delta.captured_player == 0
        assert delta.capture_count == 0
    
    def test_multiple_captures(self):
        """Test MoveDelta with multiple separate captures."""
        # In Pente, you can capture in multiple directions in one move
        captured_positions = [
            (5, 6), (5, 7),    # First pair
            (6, 5), (7, 5)     # Second pair
        ]
        
        delta = MoveDelta(
            position=(5, 5),
            player=1,
            captured_positions=captured_positions,
            captured_player=-1,
            capture_count=2  # Two pairs captured
        )
        
        assert len(delta.captured_positions) == 4
        assert delta.capture_count == 2
        assert delta.has_captures()
    
    def test_immutable_captured_positions(self):
        """Test that captured_positions can be safely modified without affecting delta."""
        original_positions = [(1, 2), (3, 4)]
        delta = MoveDelta((0, 0), 1, captured_positions=original_positions)
        
        # Modify original list
        original_positions.append((5, 6))
        
        # Delta should be unchanged
        assert len(delta.captured_positions) == 2
        assert (5, 6) not in delta.captured_positions
    
    def test_equality_and_comparison(self):
        """Test that MoveDelta objects can be compared."""
        delta1 = MoveDelta((5, 5), 1, captured_positions=[(2, 3)], capture_count=1)
        delta2 = MoveDelta((5, 5), 1, captured_positions=[(2, 3)], capture_count=1)
        delta3 = MoveDelta((5, 5), -1, captured_positions=[(2, 3)], capture_count=1)
        
        # Note: dataclass automatically provides __eq__ if eq=True (default)
        # Since we're using @dataclass, these should be equal if all fields match
        assert delta1 == delta2
        assert delta1 != delta3
    
    def test_zero_captures_vs_none_captures(self):
        """Test difference between zero captures and no capture information."""
        # No capture information (default)
        delta1 = MoveDelta((0, 0), 1)
        
        # Explicit zero captures
        delta2 = MoveDelta((0, 0), 1, captured_positions=[], capture_count=0)
        
        # Both should behave the same
        assert not delta1.has_captures()
        assert not delta2.has_captures()
        assert delta1.capture_count == delta2.capture_count
        assert len(delta1.captured_positions) == len(delta2.captured_positions)
    
    def test_edge_case_positions(self):
        """Test MoveDelta with edge case positions."""
        # Corner positions
        corners = [(0, 0), (0, 18), (18, 0), (18, 18)]
        
        for pos in corners:
            delta = MoveDelta(pos, 1)
            assert delta.position == pos
            assert delta.player == 1
    
    def test_capture_count_consistency(self):
        """Test that capture_count makes sense with captured_positions."""
        # In Pente, captures happen in pairs
        # So if we capture 4 stones, that's 2 pairs = capture_count of 2
        
        captured_positions = [(1, 1), (1, 2), (3, 3), (3, 4)]  # 4 stones
        delta = MoveDelta(
            (0, 0), 1,
            captured_positions=captured_positions,
            capture_count=2  # 2 pairs
        )
        
        assert len(delta.captured_positions) == 4
        assert delta.capture_count == 2
        assert delta.has_captures()
        
        # Typical single capture (1 pair = 2 stones)
        delta2 = MoveDelta(
            (5, 5), -1,
            captured_positions=[(3, 3), (4, 4)],
            capture_count=1
        )
        
        assert len(delta2.captured_positions) == 2
        assert delta2.capture_count == 1