import unittest
import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from games.pente import Pente

class TestPenteLegalMoves(unittest.TestCase):
    """Test cases for Pente get_legal_moves function on 7x7 board."""
    
    def setUp(self):
        """Set up test fixtures with 7x7 board."""
        self.game = Pente(board_size=7, captures_to_win=5, tournament_rule=True)
        self.center = 3  # Center of 7x7 board
    
    def test_initial_legal_moves_all_empty(self):
        """Test that initially all positions are legal moves."""
        legal_moves = self.game.get_legal_moves()
        
        # All 49 positions should be legal initially
        expected_moves = [(i, j) for i in range(7) for j in range(7)]
        self.assertEqual(len(legal_moves), 49)
        self.assertEqual(set(legal_moves), set(expected_moves))
    
    def test_tournament_rule_second_move_player_1(self):
        """Test tournament rule restricts player 1's second move."""
        # Place first move at center (player 1)
        self.game.make_move((self.center, self.center))
        
        # Place player 2's move
        self.game.make_move((2, 2))
        
        # Now it's player 1's second move (move_count = 2, current_player = 1)
        # The tournament rule checks for move_count == 2 and current_player == 1
        # This happens when player 1 is about to make their second move
        # Let's create the correct scenario
        
        # Reset and set up the exact scenario for tournament rule
        self.game = Pente(board_size=7, captures_to_win=5, tournament_rule=True)
        self.game.board[self.center, self.center] = 1  # Place player 1's first move
        self.game.board[2, 2] = -1  # Place player 2's first move
        self.game.move_history.append((self.center, self.center))
        self.game.move_history.append((2, 2))
        self.game.move_count = 2
        self.game.current_player = 1  # Player 1's turn for second move
        
        legal_moves = self.game.get_legal_moves()
        
        # Check that all legal moves are at least distance 3 from center
        for row, col in legal_moves:
            distance = max(abs(row - self.center), abs(col - self.center))
            self.assertGreaterEqual(distance, 3, 
                                  f"Move ({row}, {col}) violates tournament rule (distance {distance} < 3)")
        
        # Verify some specific positions
        # These should NOT be legal (too close to center)
        invalid_positions = [
            (1, 1), (1, 2), (1, 3), (1, 4), (1, 5),  # Distance 2 moves
            (2, 1), (2, 2), (2, 4), (2, 5),          # Distance 2 moves
            (3, 1), (3, 2), (3, 4), (3, 5),          # Distance 2 moves
            (4, 1), (4, 2), (4, 4), (4, 5),          # Distance 2 moves
            (5, 1), (5, 2), (5, 3), (5, 4), (5, 5)   # Distance 2 moves
        ]
        
        for pos in invalid_positions:
            self.assertNotIn(pos, legal_moves, 
                           f"Position {pos} should not be legal due to tournament rule")
        
        # These should be legal (distance >= 3)
        valid_positions = [
            (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),  # Top row
            (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6),  # Bottom row
            (1, 0), (2, 0), (3, 0), (4, 0), (5, 0),                  # Left column
            (1, 6), (2, 6), (3, 6), (4, 6), (5, 6)                   # Right column
        ]
        
        for pos in valid_positions:
            if self.game.board[pos] == 0:  # Only check empty positions
                self.assertIn(pos, legal_moves, 
                            f"Position {pos} should be legal under tournament rule")
    
    def test_tournament_rule_not_applied_to_player_2(self):
        """Test tournament rule doesn't apply to player 2 (player -1)."""
        # Place first move at center (player 1)
        self.game.make_move((self.center, self.center))
        
        # Place player 2's move (should not be restricted by tournament rule)
        # Current player is now -1, move_count is 1
        legal_moves = self.game.get_legal_moves()
        
        # Player 2 should be able to move anywhere except the occupied center
        expected_count = 48  # 49 total - 1 occupied center
        self.assertEqual(len(legal_moves), expected_count)
        
        # Verify player 2 can move close to center (tournament rule doesn't apply)
        close_to_center_moves = [(2, 2), (2, 3), (2, 4), (3, 2), (3, 4), (4, 2), (4, 3), (4, 4)]
        for move in close_to_center_moves:
            self.assertIn(move, legal_moves, 
                        f"Player 2 should be able to move to {move}")
    
    def test_tournament_rule_after_second_move(self):
        """Test tournament rule doesn't apply after player 1's second move."""
        # Place first move at center (player 1)
        self.game.make_move((self.center, self.center))
        
        # Place player 2's move
        self.game.make_move((2, 2))
        
        # Place player 1's second move (restricted by tournament rule)
        self.game.make_move((0, 0))
        
        # Now move_count = 3, tournament rule should no longer apply
        legal_moves = self.game.get_legal_moves()
        
        # Player 2 should be able to move anywhere except occupied positions
        expected_count = 46  # 49 total - 3 occupied
        self.assertEqual(len(legal_moves), expected_count)
        
        # Verify moves close to center are now allowed
        if self.game.board[2, 3] == 0:
            self.assertIn((2, 3), legal_moves, 
                        "Moves close to center should be allowed after second move")
    
    def test_no_tournament_rule(self):
        """Test behavior when tournament rule is disabled."""
        # Create game without tournament rule
        game_no_rule = Pente(board_size=7, captures_to_win=5, tournament_rule=False)
        
        # Place first move at center
        game_no_rule.make_move((self.center, self.center))
        
        # All remaining positions should be legal (no tournament rule restriction)
        legal_moves = game_no_rule.get_legal_moves()
        expected_count = 48  # 49 total - 1 occupied
        self.assertEqual(len(legal_moves), expected_count)
        
        # Verify moves close to center are allowed
        close_moves = [(2, 2), (2, 3), (2, 4), (3, 2), (3, 4)]
        for move in close_moves:
            self.assertIn(move, legal_moves, 
                        f"Move {move} should be legal when tournament rule is disabled")
    
    def test_occupied_positions_not_legal(self):
        """Test that occupied positions are not in legal moves."""
        # Place some stones
        moves = [(3, 3), (2, 2), (4, 4), (1, 1)]
        for move in moves:
            self.game.make_move(move)
        
        legal_moves = self.game.get_legal_moves()
        
        # Occupied positions should not be legal
        for move in moves:
            self.assertNotIn(move, legal_moves, 
                           f"Occupied position {move} should not be legal")
        
        # Total legal moves should be 49 - number of occupied positions
        expected_count = 49 - len(moves)
        self.assertEqual(len(legal_moves), expected_count)
    
    def test_board_boundaries(self):
        """Test that moves outside board boundaries are not included."""
        legal_moves = self.game.get_legal_moves()
        
        # All legal moves should be within board boundaries
        for row, col in legal_moves:
            self.assertGreaterEqual(row, 0, "Row should be >= 0")
            self.assertLess(row, 7, "Row should be < 7")
            self.assertGreaterEqual(col, 0, "Col should be >= 0")
            self.assertLess(col, 7, "Col should be < 7")
    
    def test_tournament_rule_distance_calculation(self):
        """Test specific distance calculations for tournament rule."""
        # Set up the correct scenario for tournament rule (player 1's second move)
        self.game.board[self.center, self.center] = 1  # Place player 1's first move
        self.game.board[2, 2] = -1  # Place player 2's first move
        self.game.move_history.append((self.center, self.center))
        self.game.move_history.append((2, 2))
        self.game.move_count = 2
        self.game.current_player = 1  # Player 1's turn for second move
        
        legal_moves = self.game.get_legal_moves()
        
        # Test specific positions and their distances
        test_cases = [
            # Position, Expected Distance, Should be Legal
            ((0, 0), 3, True),    # Corner, distance 3
            ((0, 3), 3, True),    # Top edge, distance 3
            ((6, 6), 3, True),    # Opposite corner, distance 3
            ((1, 1), 2, False),   # Too close, distance 2
            ((2, 3), 1, False),   # Adjacent, distance 1
            ((3, 5), 2, False),   # Horizontal distance 2
            ((5, 3), 2, False),   # Vertical distance 2
        ]
        
        for (row, col), expected_distance, should_be_legal in test_cases:
            actual_distance = max(abs(row - self.center), abs(col - self.center))
            self.assertEqual(actual_distance, expected_distance, 
                           f"Distance calculation wrong for ({row}, {col})")
            
            if should_be_legal:
                self.assertIn((row, col), legal_moves, 
                            f"Position ({row}, {col}) should be legal (distance {expected_distance})")
            else:
                self.assertNotIn((row, col), legal_moves, 
                               f"Position ({row}, {col}) should not be legal (distance {expected_distance})")
    
    def test_empty_board_legal_moves_count(self):
        """Test that empty board has exactly 49 legal moves on 7x7."""
        legal_moves = self.game.get_legal_moves()
        self.assertEqual(len(legal_moves), 49, "Empty 7x7 board should have 49 legal moves")
        
        # Verify no duplicates
        self.assertEqual(len(legal_moves), len(set(legal_moves)), 
                        "Legal moves should not contain duplicates")

if __name__ == '__main__':
    unittest.main()