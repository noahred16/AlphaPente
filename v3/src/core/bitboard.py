"""
High-performance bitboard representation for 19x19 Pente board.
Uses 6 64-bit integers per player to represent all 361 positions.
"""

from typing import List, Tuple
import math

class BitBoard:
    """
    BitBoard representation for 19x19 board using 6 64-bit integers per player.
    
    Board positions are mapped to bit indices 0-360:
    - Position (row, col) maps to bit index: row * 19 + col
    - 361 positions require 6 64-bit integers (6 * 64 = 384 bits)
    """
    
    def __init__(self):
        # 6 64-bit integers for each player (1 and -1)
        self.player1_bits = [0] * 6  # Player 1 stones
        self.player2_bits = [0] * 6  # Player -1 stones
        
    def _pos_to_bit_index(self, row: int, col: int) -> Tuple[int, int]:
        """Convert (row, col) to (chunk_index, bit_position)."""
        bit_index = row * 19 + col
        chunk_index = bit_index // 64
        bit_position = bit_index % 64
        return chunk_index, bit_position
    
    def set_stone(self, row: int, col: int, player: int) -> None:
        """Place a stone for the given player."""
        if not (0 <= row < 19 and 0 <= col < 19):
            raise ValueError(f"Invalid position: ({row}, {col})")
        if player not in (1, -1):
            raise ValueError(f"Invalid player: {player}")
            
        chunk_index, bit_position = self._pos_to_bit_index(row, col)
        bit_mask = 1 << bit_position
        
        # Clear both players first (in case position was occupied)
        self.player1_bits[chunk_index] &= ~bit_mask
        self.player2_bits[chunk_index] &= ~bit_mask
        
        # Set the appropriate player's bit
        if player == 1:
            self.player1_bits[chunk_index] |= bit_mask
        else:
            self.player2_bits[chunk_index] |= bit_mask
    
    def remove_stone(self, row: int, col: int) -> None:
        """Remove stone from the given position."""
        if not (0 <= row < 19 and 0 <= col < 19):
            raise ValueError(f"Invalid position: ({row}, {col})")
            
        chunk_index, bit_position = self._pos_to_bit_index(row, col)
        bit_mask = ~(1 << bit_position)
        
        self.player1_bits[chunk_index] &= bit_mask
        self.player2_bits[chunk_index] &= bit_mask
    
    def get_stone(self, row: int, col: int) -> int:
        """Get the player at the given position (0 if empty, 1 or -1 if occupied)."""
        if not (0 <= row < 19 and 0 <= col < 19):
            raise ValueError(f"Invalid position: ({row}, {col})")
            
        chunk_index, bit_position = self._pos_to_bit_index(row, col)
        bit_mask = 1 << bit_position
        
        if self.player1_bits[chunk_index] & bit_mask:
            return 1
        elif self.player2_bits[chunk_index] & bit_mask:
            return -1
        else:
            return 0
    
    def is_empty(self, row: int, col: int) -> bool:
        """Check if position is empty."""
        return self.get_stone(row, col) == 0
    
    def get_all_stones(self) -> List[Tuple[int, int, int]]:
        """Get all stones on the board as (row, col, player) tuples."""
        stones = []
        
        for chunk_index in range(6):
            for bit_position in range(64):
                bit_index = chunk_index * 64 + bit_position
                if bit_index >= 361:  # Only 361 positions on 19x19 board
                    break
                    
                row = bit_index // 19
                col = bit_index % 19
                
                bit_mask = 1 << bit_position
                if self.player1_bits[chunk_index] & bit_mask:
                    stones.append((row, col, 1))
                elif self.player2_bits[chunk_index] & bit_mask:
                    stones.append((row, col, -1))
        
        return stones
    
    def get_stone_positions(self, player: int) -> List[Tuple[int, int]]:
        """Get all positions occupied by the given player."""
        if player not in (1, -1):
            raise ValueError(f"Invalid player: {player}")
            
        positions = []
        bits = self.player1_bits if player == 1 else self.player2_bits
        
        for chunk_index in range(6):
            chunk = bits[chunk_index]
            for bit_position in range(64):
                if chunk & (1 << bit_position):
                    bit_index = chunk_index * 64 + bit_position
                    if bit_index >= 361:  # Only 361 positions on 19x19 board
                        break
                    row = bit_index // 19
                    col = bit_index % 19
                    positions.append((row, col))
        
        return positions
    
    def clear(self) -> None:
        """Clear the entire board."""
        self.player1_bits = [0] * 6
        self.player2_bits = [0] * 6
    
    def copy(self) -> 'BitBoard':
        """Create a copy of this bitboard."""
        new_board = BitBoard()
        new_board.player1_bits = self.player1_bits[:]
        new_board.player2_bits = self.player2_bits[:]
        return new_board
    
    def __eq__(self, other: 'BitBoard') -> bool:
        """Check equality with another bitboard."""
        if not isinstance(other, BitBoard):
            return False
        return (self.player1_bits == other.player1_bits and 
                self.player2_bits == other.player2_bits)
    
    def __str__(self) -> str:
        """String representation of the board."""
        lines = []
        for row in range(19):
            line = []
            for col in range(19):
                stone = self.get_stone(row, col)
                if stone == 1:
                    line.append('X')
                elif stone == -1:
                    line.append('O')
                else:
                    line.append('.')
            lines.append(' '.join(line))
        return '\n'.join(lines)