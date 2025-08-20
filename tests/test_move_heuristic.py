import unittest
import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from games.pente import Pente
from mcts.move_heuristic import MoveHeuristic

class TestMoveHeuristic(unittest.TestCase):
    """Test cases for move heuristic evaluation system."""
    
    def setUp(self):
        """Set up test fixtures with 7x7 board."""
        self.game = Pente(board_size=7)
        self.heuristic = MoveHeuristic(self.game)
    
    def test_critical_winning_move(self):
        """Test detection of winning move (5 in a row)."""
        # Set up a position where player 1 can win with one move
        # Four in a row horizontally: (3,1), (3,2), (3,3), (3,4)
        positions = [(3, 1), (3, 2), (3, 3), (3, 4)]
        
        for pos in positions:
            self.game.board[pos] = 1
        
        # Player 1's turn, winning move at (3, 5)
        self.game.current_player = 1
        winning_move = (3, 5)
        
        score = self.heuristic._evaluate_move(winning_move)
        self.assertEqual(score, MoveHeuristic.WEIGHTS['critical'], 
                        "Should detect winning move as critical priority")
    
    def test_critical_blocking_move(self):
        """Test detection of move that blocks opponent's win."""
        # Set up position where opponent (player -1) is about to win
        positions = [(2, 1), (2, 2), (2, 3), (2, 4)]
        
        for pos in positions:
            self.game.board[pos] = -1
        
        # Player 1's turn, must block at (2, 5)
        self.game.current_player = 1
        blocking_move = (2, 5)
        
        score = self.heuristic._evaluate_move(blocking_move)
        self.assertEqual(score, MoveHeuristic.WEIGHTS['critical'],
                        "Should detect blocking move as critical priority")
    
    def test_very_high_capture_move(self):
        """Test detection of move that creates a capture."""
        # Set up capture pattern: Player 1 at (3,1), opponent stones at (3,2) and (3,3)
        self.game.board[3, 1] = 1
        self.game.board[3, 2] = -1
        self.game.board[3, 3] = -1
        
        # Player 1's turn, capture move at (3, 4)
        self.game.current_player = 1
        capture_move = (3, 4)
        
        score = self.heuristic._evaluate_move(capture_move)
        self.assertEqual(score, MoveHeuristic.WEIGHTS['very_high'],
                        "Should detect capture move as very high priority")
    
    def test_very_high_open_four(self):
        """Test detection of move that creates open four."""
        # Set up three in a row with open ends: (2,2), (2,3), (2,4)
        positions = [(2, 2), (2, 3), (2, 4)]
        
        for pos in positions:
            self.game.board[pos] = 1
        
        # Player 1's turn, creating open four at (2, 1) or (2, 5)
        self.game.current_player = 1
        open_four_move = (2, 1)
        
        score = self.heuristic._evaluate_move(open_four_move)
        self.assertEqual(score, MoveHeuristic.WEIGHTS['very_high'],
                        "Should detect open four creation as very high priority")
    
    def test_high_open_three_creation(self):
        """Test detection of move that creates open three."""
        # Set up two in a row with open ends: (3,2), (3,3)
        self.game.board[3, 2] = 1
        self.game.board[3, 3] = 1
        
        # Player 1's turn, creating open three at (3, 1) or (3, 4)
        self.game.current_player = 1
        open_three_move = (3, 1)
        
        score = self.heuristic._evaluate_move(open_three_move)
        self.assertEqual(score, MoveHeuristic.WEIGHTS['high'],
                        "Should detect open three creation as high priority")
    
    def test_high_block_open_three(self):
        """Test detection of move that blocks opponent's open three."""
        # Set up opponent's two in a row: (4,2), (4,3)
        self.game.board[4, 2] = -1
        self.game.board[4, 3] = -1
        
        # Player 1's turn, blocking at (4, 1) or (4, 4)
        self.game.current_player = 1
        blocking_move = (4, 1)
        
        score = self.heuristic._evaluate_move(blocking_move)
        self.assertEqual(score, MoveHeuristic.WEIGHTS['high'],
                        "Should detect blocking open three as high priority")
    
    def test_medium_near_cluster(self):
        """Test detection of move near existing stones (cluster)."""
        # Place some stones to create a cluster
        self.game.board[3, 3] = 1
        self.game.board[4, 4] = -1
        
        # Player 1's turn, move near cluster (within distance 2)
        self.game.current_player = 1
        near_cluster_move = (2, 2)  # Distance 1 from (3,3)
        
        score = self.heuristic._evaluate_move(near_cluster_move)
        self.assertEqual(score, MoveHeuristic.WEIGHTS['medium'],
                        "Should detect move near cluster as medium priority")
    
    def test_low_priority_isolated_move(self):
        """Test detection of isolated move (low priority)."""
        # Empty board except for some distant stones
        self.game.board[0, 0] = 1
        self.game.board[6, 6] = -1
        
        # Player 1's turn, isolated move
        self.game.current_player = 1
        isolated_move = (3, 3)
        
        score = self.heuristic._evaluate_move(isolated_move)
        self.assertEqual(score, MoveHeuristic.WEIGHTS['low'],
                        "Should detect isolated move as low priority")
    
    def test_move_evaluation_ordering(self):
        """Test that moves are properly ordered by priority."""
        # Set up a complex position with multiple move types
        
        # Critical: Opponent about to win
        self.game.board[1, 1] = -1
        self.game.board[1, 2] = -1
        self.game.board[1, 3] = -1
        self.game.board[1, 4] = -1
        # Block at (1, 5) - critical
        
        # Very high: Capture opportunity
        self.game.board[3, 1] = 1
        self.game.board[3, 2] = -1
        self.game.board[3, 3] = -1
        # Capture at (3, 4) - very high
        
        # High: Create open three
        self.game.board[5, 2] = 1
        self.game.board[5, 3] = 1
        # Open three at (5, 1) or (5, 4) - high
        
        # Medium: Near cluster
        # Move at (4, 2) near existing stones - medium
        
        # Low: Isolated
        # Move at (0, 6) - low
        
        moves = [(1, 5), (3, 4), (5, 1), (4, 2), (0, 6)]
        self.game.current_player = 1
        
        evaluated_moves = self.heuristic.evaluate_moves(moves)
        
        # Check that moves are ordered by priority (highest first)
        expected_order = [(1, 5), (3, 4), (5, 1), (4, 2), (0, 6)]
        actual_order = [move for move, score in evaluated_moves]
        
        self.assertEqual(actual_order, expected_order,
                        "Moves should be ordered by heuristic priority")
    
    def test_tournament_rule_second_move(self):
        """Test that tournament rule is respected for player 1's second move."""
        # Place first move at center
        center = self.game.board_size // 2
        self.game.board[center, center] = 1
        self.game.board[2, 2] = -1  # Player 2's move
        self.game.move_count = 2
        self.game.current_player = 1
        
        # Get legal moves for second move
        legal_moves = self.game.get_legal_moves()
        
        # All legal moves should be at least 3 intersections from center
        for move in legal_moves:
            row, col = move
            distance = max(abs(row - center), abs(col - center))
            self.assertGreaterEqual(distance, 3,
                                  f"Move {move} violates tournament rule (distance {distance} < 3)")
    
    def test_capture_win_condition(self):
        """Test detection of winning by captures."""
        # Set up game state where player 1 needs one more capture to win
        self.game.captures[1] = 4  # Need 5 to win, so 4 captured pairs = 8 stones
        
        # Set up capture opportunity
        self.game.board[2, 1] = 1
        self.game.board[2, 2] = -1
        self.game.board[2, 3] = -1
        
        self.game.current_player = 1
        winning_capture = (2, 4)
        
        score = self.heuristic._evaluate_move(winning_capture)
        self.assertEqual(score, MoveHeuristic.WEIGHTS['critical'],
                        "Should detect capture that wins game as critical")

if __name__ == '__main__':
    unittest.main()