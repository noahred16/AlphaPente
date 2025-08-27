import unittest
import sys
import os
import math

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from games.pente import Pente
from mcts.mcts import MCTS
from players.mcts_player import MCTSPlayer


class TestScaledValuation(unittest.TestCase):
    """Test the new move-count-based scaled valuation system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mcts = MCTS(max_iterations=50)
        
    def test_valuation_formula_basic(self):
        """Test the basic valuation formula calculations."""
        # Test tie scenarios
        tie_value = self.mcts._calculate_scaled_valuation(None, 1, 10)
        self.assertEqual(tie_value, 0.0, "Ties should always be 0.0")
        
        tie_value_long = self.mcts._calculate_scaled_valuation(None, 1, 100)
        self.assertEqual(tie_value_long, 0.0, "Ties should always be 0.0 regardless of move count")
        
    def test_valuation_win_scaling(self):
        """Test that wins scale correctly based on move count."""
        # Fast win (few moves) should have high value
        fast_win = self.mcts._calculate_scaled_valuation(1, 1, 5)
        self.assertGreater(fast_win, 0.8, "Fast wins should have high positive value")
        self.assertLessEqual(fast_win, 1.0, "Win values should not exceed 1.0")
        
        # Medium-speed win
        medium_win = self.mcts._calculate_scaled_valuation(1, 1, 25)
        self.assertGreater(medium_win, 0.4, "Medium wins should have moderate positive value")
        self.assertLess(medium_win, fast_win, "Medium wins should be less valuable than fast wins")
        
        # Slow win (many moves) should have lower value
        slow_win = self.mcts._calculate_scaled_valuation(1, 1, 100)
        self.assertGreater(slow_win, 0.2, "Slow wins should still be positive")
        self.assertLess(slow_win, medium_win, "Slow wins should be less valuable than medium wins")
        
    def test_valuation_loss_scaling(self):
        """Test that losses scale correctly based on move count."""
        # Fast loss (few moves) should have very negative value
        fast_loss = self.mcts._calculate_scaled_valuation(-1, 1, 5)
        self.assertLess(fast_loss, -0.8, "Fast losses should have high negative value")
        self.assertGreaterEqual(fast_loss, -1.0, "Loss values should not be less than -1.0")
        
        # Medium-speed loss
        medium_loss = self.mcts._calculate_scaled_valuation(-1, 1, 25)
        self.assertLess(medium_loss, -0.4, "Medium losses should have moderate negative value")
        self.assertGreater(medium_loss, fast_loss, "Medium losses should be less severe than fast losses")
        
        # Slow loss (many moves) should have less negative value
        slow_loss = self.mcts._calculate_scaled_valuation(-1, 1, 100)
        self.assertLess(slow_loss, -0.2, "Slow losses should still be negative")
        self.assertGreater(slow_loss, medium_loss, "Slow losses should be less severe than medium losses")
        
    def test_valuation_symmetry(self):
        """Test that win/loss values are symmetric for the same move count."""
        for move_count in [5, 10, 25, 50, 100]:
            win_value = self.mcts._calculate_scaled_valuation(1, 1, move_count)
            loss_value = self.mcts._calculate_scaled_valuation(-1, 1, move_count)
            
            self.assertAlmostEqual(win_value, -loss_value, places=6,
                                 msg=f"Win and loss values should be symmetric for {move_count} moves")
            
    def test_valuation_bounds(self):
        """Test that valuations stay within expected bounds."""
        # Test various move counts
        for move_count in range(1, 201, 10):
            win_value = self.mcts._calculate_scaled_valuation(1, 1, move_count)
            loss_value = self.mcts._calculate_scaled_valuation(-1, 1, move_count)
            
            # Check bounds
            self.assertGreaterEqual(win_value, 0.2, f"Win value at {move_count} moves should be >= 0.2")
            self.assertLessEqual(win_value, 1.0, f"Win value at {move_count} moves should be <= 1.0")
            
            self.assertLessEqual(loss_value, -0.2, f"Loss value at {move_count} moves should be <= -0.2")
            self.assertGreaterEqual(loss_value, -1.0, f"Loss value at {move_count} moves should be >= -1.0")
            
    def test_valuation_monotonicity(self):
        """Test that valuation decreases monotonically with move count."""
        # Test that longer games have lower absolute values
        move_counts = [5, 10, 20, 30, 50, 100]
        win_values = []
        loss_values = []
        
        for count in move_counts:
            win_values.append(self.mcts._calculate_scaled_valuation(1, 1, count))
            loss_values.append(self.mcts._calculate_scaled_valuation(-1, 1, count))
            
        # Check that win values decrease with more moves
        for i in range(1, len(win_values)):
            self.assertLess(win_values[i], win_values[i-1],
                           f"Win value should decrease from {move_counts[i-1]} to {move_counts[i]} moves")
            
        # Check that loss values become less negative with more moves
        for i in range(1, len(loss_values)):
            self.assertGreater(loss_values[i], loss_values[i-1],
                             f"Loss value should become less negative from {move_counts[i-1]} to {move_counts[i]} moves")
            
    def test_perspective_handling(self):
        """Test that perspective is handled correctly."""
        # Player 1 wins
        player1_win_from_p1 = self.mcts._calculate_scaled_valuation(1, 1, 20)
        player1_win_from_p2 = self.mcts._calculate_scaled_valuation(1, -1, 20)
        
        self.assertGreater(player1_win_from_p1, 0, "Player 1 win should be positive from Player 1's perspective")
        self.assertLess(player1_win_from_p2, 0, "Player 1 win should be negative from Player 2's perspective")
        self.assertAlmostEqual(player1_win_from_p1, -player1_win_from_p2, places=6,
                             msg="Win should be symmetric from different perspectives")
        
    def test_integration_with_mcts(self):
        """Test that the new valuation system works with full MCTS."""
        game = Pente(board_size=7, tournament_rule=False)
        game.current_player = 1
        
        # Run a short MCTS search
        mcts = MCTS(max_iterations=20)
        move = mcts.search(game)
        
        # Should return a valid move
        self.assertIsInstance(move, tuple, "MCTS should return a move tuple")
        self.assertEqual(len(move), 2, "Move should be a coordinate pair")
        
        legal_moves = game.get_legal_moves()
        self.assertIn(move, legal_moves, "MCTS should return a legal move")
        
    def test_mcts_player_integration(self):
        """Test that MCTSPlayer works with the new valuation system."""
        game = Pente(board_size=7, tournament_rule=False)
        player = MCTSPlayer(name="Test AI", player_id=1, max_iterations=30)
        
        # Player should be able to make moves
        move = player.get_move(game)
        self.assertIsInstance(move, tuple, "Player should return a move tuple")
        
        legal_moves = game.get_legal_moves()
        self.assertIn(move, legal_moves, "Player should return a legal move")
        
    def test_tactical_preference_fast_wins(self):
        """Test that MCTS prefers faster wins when given the choice."""
        # Create a position where MCTS can win in different numbers of moves
        game = Pente(board_size=7, tournament_rule=False)
        
        # Set up a position where player can win immediately
        # Four stones in a row, need one more to win
        game.board[3, 1] = 1
        game.board[3, 2] = 1  
        game.board[3, 3] = 1
        game.board[3, 4] = 1  # Can win at (3,0) or (3,5)
        game.current_player = 1
        
        mcts = MCTS(max_iterations=100)  # More iterations for reliable results
        move = mcts.search(game)
        
        # Should choose a winning move
        winning_moves = [(3, 0), (3, 5)]
        self.assertIn(move, winning_moves, "MCTS should choose a winning move when available")
        
    def test_statistical_consistency(self):
        """Test that the valuation system produces consistent statistics."""
        game = Pente(board_size=5, tournament_rule=False)
        mcts = MCTS(max_iterations=50)
        
        # Run multiple searches to get statistics
        stats = mcts.get_move_statistics(game)
        
        # Should have explored some moves
        self.assertGreater(len(stats), 0, "Should have move statistics")
        
        # Check that statistics are in reasonable range
        for move, visits, value in stats:
            self.assertIsInstance(move, tuple, "Move should be tuple")
            self.assertGreater(visits, 0, "Visits should be positive")
            self.assertGreaterEqual(value, -1.0, "Value should be >= -1.0")
            self.assertLessEqual(value, 1.0, "Value should be <= 1.0")
            
    def test_extreme_move_counts(self):
        """Test valuation with extreme move counts."""
        # Very fast game (1 move)
        very_fast_win = self.mcts._calculate_scaled_valuation(1, 1, 1)
        self.assertGreater(very_fast_win, 0.95, "1-move win should be very high value")
        
        # Very long game (500 moves)
        very_slow_win = self.mcts._calculate_scaled_valuation(1, 1, 500)
        self.assertLess(very_slow_win, 0.25, "500-move win should be low value")
        self.assertGreater(very_slow_win, 0.15, "500-move win should still be positive")


class TestScaledValuationBehavior(unittest.TestCase):
    """Test behavioral aspects of the scaled valuation system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.board_size = 7
        self.player = MCTSPlayer(name="Test AI", player_id=1, max_iterations=50)
        
    def test_aggressive_vs_safe_play(self):
        """Test that the AI shows appropriate aggression vs safety preferences."""
        # This is a behavioral test - we expect the AI to be more aggressive
        # in pursuing quick wins and more defensive against quick losses
        
        game = Pente(board_size=self.board_size, tournament_rule=False)
        
        # Test that the AI can still make reasonable moves
        for _ in range(5):
            if not game.is_terminal():
                move = self.player.get_move(game)
                legal_moves = game.get_legal_moves()
                self.assertIn(move, legal_moves, "AI should make legal moves")
                
                # Make the move to continue the game
                game.make_move(move)
                game.current_player *= -1  # Switch players for test
                
    def test_preference_consistency(self):
        """Test that move preferences are consistent with valuation goals."""
        # Create multiple identical positions and ensure consistent preferences
        game = Pente(board_size=5, tournament_rule=False)
        game.current_player = 1
        
        moves = []
        for _ in range(3):
            move = self.player.get_move(game)
            moves.append(move)
            
        # While not requiring identical moves (due to randomness in MCTS),
        # the moves should all be legal
        legal_moves = game.get_legal_moves()
        for move in moves:
            self.assertIn(move, legal_moves, "All moves should be legal")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)