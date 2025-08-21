import unittest
import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from games.pente import Pente
from mcts.move_heuristic import MoveHeuristic


class TestDiagonalHeuristics(unittest.TestCase):
    """Test heuristic evaluation specifically for diagonal patterns."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.board_size = 7
        
    def _print_heuristic_analysis(self, game, test_name):
        """Helper to print detailed heuristic analysis."""
        print(f"\n=== {test_name} ===")
        print("Board:")
        for i in range(game.board_size):
            row_str = f"{i}: "
            for j in range(game.board_size):
                if game.board[i, j] == 1:
                    row_str += "X "
                elif game.board[i, j] == -1:
                    row_str += "O "
                else:
                    row_str += ". "
            print(row_str)
        
        heuristic = MoveHeuristic(game)
        legal_moves = game.get_legal_moves()
        evaluated_moves = heuristic.evaluate_moves(legal_moves)
        
        print(f"\nTop 10 moves by heuristic score:")
        for i, (move, score) in enumerate(evaluated_moves[:10]):
            print(f"{i+1:2}. {move} = {score:3.0f}")
            
        return heuristic, evaluated_moves
    
    def test_diagonal_winning_detection_backslash(self):
        """Test detection of diagonal winning moves (\\ direction)."""
        game = Pente(board_size=self.board_size, tournament_rule=False)
        
        # Create diagonal pattern: (1,1), (2,2), (3,3), (4,4)
        diagonal_positions = [(1, 1), (2, 2), (3, 3), (4, 4)]
        for row, col in diagonal_positions:
            game.board[row, col] = 1
        game.current_player = 1
        
        heuristic, evaluated_moves = self._print_heuristic_analysis(game, "Diagonal Winning Detection (\\)")
        
        # Expected winning moves: (0,0), (5,5) - (6,6) doesn't complete the line
        expected_wins = [(0, 0), (5, 5)]
        
        # Check that winning moves get critical score
        for move in expected_wins:
            if move in game.get_legal_moves():
                score = heuristic._evaluate_move(move)
                is_winning = heuristic._is_winning_move(move, 1)
                print(f"Move {move}: score={score}, is_winning={is_winning}")
                self.assertEqual(score, 100, f"Move {move} should be critical (100), got {score}")
                self.assertTrue(is_winning, f"Move {move} should be detected as winning")
    
    def test_diagonal_winning_detection_slash(self):
        """Test detection of diagonal winning moves (/ direction)."""
        game = Pente(board_size=self.board_size, tournament_rule=False)
        
        # Create diagonal pattern: (4,1), (3,2), (2,3), (1,4)
        diagonal_positions = [(4, 1), (3, 2), (2, 3), (1, 4)]
        for row, col in diagonal_positions:
            game.board[row, col] = 1
        game.current_player = 1
        
        heuristic, evaluated_moves = self._print_heuristic_analysis(game, "Diagonal Winning Detection (/)")
        
        # Expected winning moves: (5,0), (0,5) - (6,0) doesn't complete the line
        expected_wins = [(5, 0), (0, 5)]
        
        # Check that winning moves get critical score
        for move in expected_wins:
            if move in game.get_legal_moves():
                score = heuristic._evaluate_move(move)
                is_winning = heuristic._is_winning_move(move, 1)
                print(f"Move {move}: score={score}, is_winning={is_winning}")
                self.assertEqual(score, 100, f"Move {move} should be critical (100), got {score}")
                self.assertTrue(is_winning, f"Move {move} should be detected as winning")
    
    def test_diagonal_blocking_detection_backslash(self):
        """Test detection of diagonal blocking moves (\\ direction)."""
        game = Pente(board_size=self.board_size, tournament_rule=False)
        
        # Create opponent diagonal pattern: (1,1), (2,2), (3,3), (4,4)
        diagonal_positions = [(1, 1), (2, 2), (3, 3), (4, 4)]
        for row, col in diagonal_positions:
            game.board[row, col] = -1
        game.current_player = 1
        
        heuristic, evaluated_moves = self._print_heuristic_analysis(game, "Diagonal Blocking Detection (\\)")
        
        # Expected blocking moves: (0,0), (5,5) - (6,6) doesn't complete the line
        expected_blocks = [(0, 0), (5, 5)]
        
        # Check that blocking moves get critical score
        for move in expected_blocks:
            if move in game.get_legal_moves():
                score = heuristic._evaluate_move(move)
                is_blocking = heuristic._is_blocking_win(move, -1)
                print(f"Move {move}: score={score}, is_blocking={is_blocking}")
                self.assertEqual(score, 100, f"Move {move} should be critical (100), got {score}")
                self.assertTrue(is_blocking, f"Move {move} should be detected as blocking")
    
    def test_diagonal_blocking_detection_slash(self):
        """Test detection of diagonal blocking moves (/ direction)."""
        game = Pente(board_size=self.board_size, tournament_rule=False)
        
        # Create opponent diagonal pattern: (4,1), (3,2), (2,3), (1,4)
        diagonal_positions = [(4, 1), (3, 2), (2, 3), (1, 4)]
        for row, col in diagonal_positions:
            game.board[row, col] = -1
        game.current_player = 1
        
        heuristic, evaluated_moves = self._print_heuristic_analysis(game, "Diagonal Blocking Detection (/)")
        
        # Expected blocking moves: (5,0), (0,5) - (6,0) doesn't complete the line
        expected_blocks = [(5, 0), (0, 5)]
        
        # Check that blocking moves get critical score
        for move in expected_blocks:
            if move in game.get_legal_moves():
                score = heuristic._evaluate_move(move)
                is_blocking = heuristic._is_blocking_win(move, -1)
                print(f"Move {move}: score={score}, is_blocking={is_blocking}")
                self.assertEqual(score, 100, f"Move {move} should be critical (100), got {score}")
                self.assertTrue(is_blocking, f"Move {move} should be detected as blocking")
    
    def test_broken_diagonal_pattern(self):
        """Test that broken diagonal patterns are not detected as critical."""
        game = Pente(board_size=self.board_size, tournament_rule=False)
        
        # Create broken diagonal: (1,1), (3,3), (4,4) - missing (2,2)
        broken_positions = [(1, 1), (3, 3), (4, 4)]
        for row, col in broken_positions:
            game.board[row, col] = 1
        game.current_player = 1
        
        heuristic, evaluated_moves = self._print_heuristic_analysis(game, "Broken Diagonal Pattern")
        
        # These should NOT be winning moves since pattern is broken
        potential_completions = [(0, 0), (5, 5)]
        
        for move in potential_completions:
            if move in game.get_legal_moves():
                score = heuristic._evaluate_move(move)
                is_winning = heuristic._is_winning_move(move, 1)
                print(f"Move {move}: score={score}, is_winning={is_winning}")
                self.assertFalse(is_winning, f"Move {move} should NOT be winning for broken pattern")
                self.assertLess(score, 100, f"Move {move} should not get critical score for broken pattern")
    
    def test_partial_diagonal_patterns(self):
        """Test heuristic evaluation of partial diagonal patterns."""
        game = Pente(board_size=self.board_size, tournament_rule=False)
        
        # Create three-in-diagonal: (2,2), (3,3), (4,4)
        three_diagonal = [(2, 2), (3, 3), (4, 4)]
        for row, col in three_diagonal:
            game.board[row, col] = 1
        game.current_player = 1
        
        heuristic, evaluated_moves = self._print_heuristic_analysis(game, "Partial Diagonal (3 in a row)")
        
        # Extensions should get high priority for creating open four or open three
        extensions = [(1, 1), (5, 5)]
        
        for move in extensions:
            if move in game.get_legal_moves():
                score = heuristic._evaluate_move(move)
                print(f"Move {move}: score={score}")
                # Should be high priority (creating open four) but not critical
                self.assertGreaterEqual(score, 20, f"Move {move} should get high priority for extending diagonal")
                self.assertLess(score, 100, f"Move {move} should not be critical (not immediate win)")
    
    def test_diagonal_vs_horizontal_priority(self):
        """Test relative priority between diagonal and horizontal patterns."""
        game = Pente(board_size=self.board_size, tournament_rule=False)
        
        # Create diagonal four for MCTS: (1,1), (2,2), (3,3), (4,4)
        diagonal_positions = [(1, 1), (2, 2), (3, 3), (4, 4)]
        for row, col in diagonal_positions:
            game.board[row, col] = 1
            
        # Create horizontal four for opponent: (5,1), (5,2), (5,3), (5,4)
        horizontal_positions = [(5, 1), (5, 2), (5, 3), (5, 4)]
        for row, col in horizontal_positions:
            game.board[row, col] = -1
            
        game.current_player = 1
        
        heuristic, evaluated_moves = self._print_heuristic_analysis(game, "Diagonal vs Horizontal Priority")
        
        # Both patterns should get critical priority
        diagonal_wins = [(0, 0), (5, 5)]  # Only these complete the diagonal
        horizontal_blocks = [(5, 0), (5, 5)]
        
        critical_moves = []
        for move in diagonal_wins + horizontal_blocks:
            if move in game.get_legal_moves():
                score = heuristic._evaluate_move(move)
                if score == 100:
                    critical_moves.append(move)
                print(f"Move {move}: score={score}")
        
        # Both winning and blocking moves should be critical
        self.assertGreater(len(critical_moves), 0, "Should have at least one critical move")
        print(f"Critical moves found: {critical_moves}")
    
    def test_edge_diagonal_patterns(self):
        """Test diagonal patterns near board edges."""
        game = Pente(board_size=self.board_size, tournament_rule=False)
        
        # Create a diagonal that CAN be extended: (1,2), (2,3), (3,4), (4,5)
        # This can be completed at (0,1) or (5,6)
        edge_diagonal = [(1, 2), (2, 3), (3, 4), (4, 5)]
        for row, col in edge_diagonal:
            game.board[row, col] = -1
        game.current_player = 1
        
        heuristic, evaluated_moves = self._print_heuristic_analysis(game, "Edge Diagonal Pattern")
        
        # Check for blocking moves
        potential_blocks = [(0, 1), (5, 6)]
        blocking_found = False
        
        for move in potential_blocks:
            if move in game.get_legal_moves():
                score = heuristic._evaluate_move(move)
                print(f"Blocking move {move}: score={score}")
                if score == 100:  # Critical blocking move
                    blocking_found = True
        
        # Should find at least one critical blocking move
        self.assertTrue(blocking_found, "Should detect critical blocking moves for completeable diagonal threat")
    
    def test_overlapping_diagonal_patterns(self):
        """Test scenario with overlapping diagonal patterns."""
        game = Pente(board_size=self.board_size, tournament_rule=False)
        
        # Create overlapping diagonals sharing center point (3,3)
        # Pattern 1: (1,1), (2,2), (3,3), (4,4) - backslash
        # Pattern 2: (1,5), (2,4), (3,3), (4,2) - slash  
        game.board[1, 1] = 1
        game.board[2, 2] = 1
        game.board[3, 3] = 1  # Shared center
        game.board[4, 4] = 1
        
        game.board[1, 5] = -1
        game.board[2, 4] = -1
        # (3,3) already occupied by player 1
        game.board[4, 2] = -1
        
        game.current_player = 1
        
        heuristic, evaluated_moves = self._print_heuristic_analysis(game, "Overlapping Diagonal Patterns")
        
        # Player 1 can win with backslash diagonal
        win_moves = [(0, 0), (5, 5)]
        # Player 1 should block opponent's slash diagonal  
        block_moves = [(5, 1)]  # (0,6) is valid position but check if it blocks
        
        critical_count = 0
        for move in win_moves + block_moves:
            if move in game.get_legal_moves():
                score = heuristic._evaluate_move(move)
                print(f"Move {move}: score={score}")
                if score == 100:
                    critical_count += 1
        
        self.assertGreater(critical_count, 0, "Should detect critical moves in overlapping patterns")


if __name__ == '__main__':
    unittest.main(verbosity=2)