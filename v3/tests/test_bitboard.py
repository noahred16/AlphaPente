"""
Comprehensive tests for BitBoard class.
"""

import pytest
from src.core.bitboard import BitBoard

class TestBitBoard:
    """Test suite for BitBoard functionality."""
    
    def test_initialization(self):
        """Test BitBoard initialization."""
        board = BitBoard()
        assert len(board.player1_bits) == 6
        assert len(board.player2_bits) == 6
        assert all(chunk == 0 for chunk in board.player1_bits)
        assert all(chunk == 0 for chunk in board.player2_bits)
    
    def test_pos_to_bit_index(self):
        """Test position to bit index conversion."""
        board = BitBoard()
        
        # Test corner positions
        assert board._pos_to_bit_index(0, 0) == (0, 0)  # First position (bit index 0)
        assert board._pos_to_bit_index(18, 18) == (5, 40)  # Last position (bit index 360: 360//64=5, 360%64=40)
        
        # Test middle position  
        assert board._pos_to_bit_index(9, 9) == (2, 52)  # Position 180 (180//64=2, 180%64=52)
        
        # Test chunk boundaries
        assert board._pos_to_bit_index(3, 6) == (0, 63)  # Position 63 (last in first chunk)
        assert board._pos_to_bit_index(3, 7) == (1, 0)   # Position 64 (first in second chunk)
    
    def test_set_and_get_stone_player1(self):
        """Test setting and getting stones for player 1."""
        board = BitBoard()
        
        # Set stone for player 1
        board.set_stone(5, 10, 1)
        assert board.get_stone(5, 10) == 1
        assert not board.is_empty(5, 10)
        
        # Test other positions remain empty
        assert board.get_stone(5, 11) == 0
        assert board.is_empty(5, 11)
    
    def test_set_and_get_stone_player2(self):
        """Test setting and getting stones for player -1."""
        board = BitBoard()
        
        # Set stone for player -1
        board.set_stone(7, 3, -1)
        assert board.get_stone(7, 3) == -1
        assert not board.is_empty(7, 3)
        
        # Test other positions remain empty
        assert board.get_stone(7, 4) == 0
        assert board.is_empty(7, 4)
    
    def test_overwrite_stone(self):
        """Test overwriting stones."""
        board = BitBoard()
        
        # Set player 1 stone
        board.set_stone(10, 10, 1)
        assert board.get_stone(10, 10) == 1
        
        # Overwrite with player -1 stone
        board.set_stone(10, 10, -1)
        assert board.get_stone(10, 10) == -1
        
        # Overwrite back to player 1
        board.set_stone(10, 10, 1)
        assert board.get_stone(10, 10) == 1
    
    def test_remove_stone(self):
        """Test removing stones."""
        board = BitBoard()
        
        # Set stones for both players
        board.set_stone(8, 8, 1)
        board.set_stone(8, 9, -1)
        
        # Remove player 1 stone
        board.remove_stone(8, 8)
        assert board.get_stone(8, 8) == 0
        assert board.is_empty(8, 8)
        
        # Player -1 stone should remain
        assert board.get_stone(8, 9) == -1
        
        # Remove player -1 stone
        board.remove_stone(8, 9)
        assert board.get_stone(8, 9) == 0
        assert board.is_empty(8, 9)
    
    def test_boundary_positions(self):
        """Test all boundary positions on 19x19 board."""
        board = BitBoard()
        
        # Test all four corners
        corners = [(0, 0), (0, 18), (18, 0), (18, 18)]
        for i, (row, col) in enumerate(corners):
            player = 1 if i % 2 == 0 else -1
            board.set_stone(row, col, player)
            assert board.get_stone(row, col) == player
    
    def test_invalid_positions(self):
        """Test invalid position handling."""
        board = BitBoard()
        
        # Test negative positions
        with pytest.raises(ValueError, match="Invalid position"):
            board.set_stone(-1, 0, 1)
        
        with pytest.raises(ValueError, match="Invalid position"):
            board.get_stone(0, -1)
        
        # Test positions beyond board
        with pytest.raises(ValueError, match="Invalid position"):
            board.set_stone(19, 0, 1)
        
        with pytest.raises(ValueError, match="Invalid position"):
            board.get_stone(0, 19)
    
    def test_invalid_player(self):
        """Test invalid player handling."""
        board = BitBoard()
        
        with pytest.raises(ValueError, match="Invalid player"):
            board.set_stone(5, 5, 2)
        
        with pytest.raises(ValueError, match="Invalid player"):
            board.set_stone(5, 5, 0)
    
    def test_get_all_stones(self):
        """Test getting all stones from board."""
        board = BitBoard()
        
        # Empty board
        assert board.get_all_stones() == []
        
        # Add some stones
        board.set_stone(0, 0, 1)
        board.set_stone(5, 10, -1)
        board.set_stone(18, 18, 1)
        
        stones = board.get_all_stones()
        expected = [(0, 0, 1), (5, 10, -1), (18, 18, 1)]
        
        # Sort both lists for comparison since order might vary
        stones.sort()
        expected.sort()
        assert stones == expected
    
    def test_get_stone_positions(self):
        """Test getting positions for specific player."""
        board = BitBoard()
        
        # Add stones for both players
        board.set_stone(1, 1, 1)
        board.set_stone(2, 2, 1)
        board.set_stone(3, 3, -1)
        board.set_stone(4, 4, -1)
        
        # Test player 1 positions
        p1_positions = board.get_stone_positions(1)
        p1_positions.sort()
        assert p1_positions == [(1, 1), (2, 2)]
        
        # Test player -1 positions
        p2_positions = board.get_stone_positions(-1)
        p2_positions.sort()
        assert p2_positions == [(3, 3), (4, 4)]
        
        # Test invalid player
        with pytest.raises(ValueError, match="Invalid player"):
            board.get_stone_positions(2)
    
    def test_clear(self):
        """Test clearing the board."""
        board = BitBoard()
        
        # Add some stones
        board.set_stone(5, 5, 1)
        board.set_stone(10, 10, -1)
        assert len(board.get_all_stones()) == 2
        
        # Clear board
        board.clear()
        assert len(board.get_all_stones()) == 0
        assert all(chunk == 0 for chunk in board.player1_bits)
        assert all(chunk == 0 for chunk in board.player2_bits)
    
    def test_copy(self):
        """Test copying bitboard."""
        board = BitBoard()
        
        # Add some stones
        board.set_stone(7, 8, 1)
        board.set_stone(12, 15, -1)
        
        # Create copy
        board_copy = board.copy()
        
        # Verify copy has same stones
        assert board_copy.get_stone(7, 8) == 1
        assert board_copy.get_stone(12, 15) == -1
        
        # Verify independence
        board.set_stone(0, 0, 1)
        assert board.get_stone(0, 0) == 1
        assert board_copy.get_stone(0, 0) == 0
    
    def test_equality(self):
        """Test bitboard equality."""
        board1 = BitBoard()
        board2 = BitBoard()
        
        # Empty boards should be equal
        assert board1 == board2
        
        # Add same stones to both
        board1.set_stone(9, 9, 1)
        board2.set_stone(9, 9, 1)
        assert board1 == board2
        
        # Make them different
        board1.set_stone(10, 10, -1)
        assert board1 != board2
        
        # Test with non-BitBoard
        assert board1 != "not a bitboard"
    
    def test_string_representation(self):
        """Test string representation."""
        board = BitBoard()
        
        # Test empty board (first few lines)
        board_str = str(board)
        lines = board_str.split('\n')
        assert len(lines) == 19  # 19x19 board
        assert all(line.count('.') == 19 for line in lines)  # All positions empty
        
        # Add some stones and test representation
        board.set_stone(0, 0, 1)    # X
        board.set_stone(0, 1, -1)   # O
        
        board_str = str(board)
        first_line = board_str.split('\n')[0]
        assert first_line.startswith('X O')
    
    def test_large_scale_operations(self):
        """Test performance with many operations."""
        board = BitBoard()
        
        # Fill half the board with alternating players
        for i in range(19):
            for j in range(10):  # Half the columns
                player = 1 if (i + j) % 2 == 0 else -1
                board.set_stone(i, j, player)
        
        # Verify count
        all_stones = board.get_all_stones()
        assert len(all_stones) == 19 * 10  # 190 stones
        
        # Count by player
        player1_count = len([s for s in all_stones if s[2] == 1])
        player2_count = len([s for s in all_stones if s[2] == -1])
        assert player1_count + player2_count == 190
    
    def test_chunk_boundaries(self):
        """Test operations across chunk boundaries."""
        board = BitBoard()
        
        # Test positions around chunk boundaries
        # Chunk 0: positions 0-63 (bit indices 0-63)
        # Chunk 1: positions 64-127 (bit indices 64-127)
        
        # Position 63 should be in chunk 0
        board.set_stone(3, 7, 1)  # bit index 3*19+7 = 64, wait that's wrong...
        # Let me recalculate: 3*19+7 = 57+7 = 64, so chunk 1, bit 0
        # Position (3,6) = 3*19+6 = 63, so chunk 0, bit 63
        
        board.set_stone(3, 6, 1)   # bit index 63 (last in chunk 0)
        board.set_stone(3, 7, -1)  # bit index 64 (first in chunk 1)
        
        assert board.get_stone(3, 6) == 1
        assert board.get_stone(3, 7) == -1
        
        # Remove and verify
        board.remove_stone(3, 6)
        board.remove_stone(3, 7)
        assert board.is_empty(3, 6)
        assert board.is_empty(3, 7)