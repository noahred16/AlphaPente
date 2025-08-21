import unittest
import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from games.pente import Pente
from players.mcts_player import MCTSPlayer


class TestAIBlockingIntegration(unittest.TestCase):
    """Integration tests for AI blocking ability in realistic game scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.board_size = 9  # Standard smaller board
        self.ai_iterations = 100  # Test with 100 simulations
        
    def _create_game_scenario(self, scenario_name: str, setup_moves: list, current_player: int = 1):
        """
        Create a game scenario from a sequence of moves.
        
        Args:
            scenario_name: Name for debugging
            setup_moves: List of (row, col, player) tuples
            current_player: Which player's turn it is (1 or -1)
        
        Returns:
            Configured Pente game
        """
        game = Pente(board_size=self.board_size, tournament_rule=False)
        
        print(f"\n=== {scenario_name} ===")
        
        # Apply setup moves
        for row, col, player in setup_moves:
            game.board[row, col] = player
            
        game.current_player = current_player
        
        # Print board state
        print("Board position:")
        for i in range(self.board_size):
            row_str = f"{i}: "
            for j in range(self.board_size):
                if game.board[i, j] == 1:
                    row_str += "X "
                elif game.board[i, j] == -1:
                    row_str += "O "
                else:
                    row_str += ". "
            print(row_str)
        
        return game
    
    def _test_ai_response(self, game: Pente, expected_moves: list, test_description: str):
        """
        Test AI's response to a position.
        
        Args:
            game: Game position to test
            expected_moves: List of acceptable moves
            test_description: Description of what AI should do
        
        Returns:
            Whether AI made an acceptable move
        """
        ai_player = MCTSPlayer(name="Test AI", player_id=game.current_player, 
                              max_iterations=self.ai_iterations)
        
        print(f"\nExpected: {test_description}")
        print(f"Acceptable moves: {expected_moves}")
        
        # Get AI move
        ai_move = ai_player.get_move(game)
        is_correct = ai_move in expected_moves
        
        print(f"AI chose: {ai_move} - {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
        
        return is_correct, ai_move
    
    def test_horizontal_threat_blocking(self):
        """Test AI blocks horizontal four-in-a-row threats."""
        # Opponent has four stones horizontally
        setup_moves = [
            (4, 2, -1),  # Opponent stones
            (4, 3, -1),
            (4, 4, -1),
            (4, 5, -1),
            (2, 2, 1),   # Some AI stones elsewhere
            (3, 3, 1),
        ]
        
        game = self._create_game_scenario("Horizontal Threat", setup_moves, current_player=1)
        
        # AI must block at (4,1) or (4,6)
        expected_blocks = [(4, 1), (4, 6)]
        is_correct, move = self._test_ai_response(game, expected_blocks, "Block horizontal threat")
        
        self.assertTrue(is_correct, f"AI should block horizontal threat, chose {move}")
    
    def test_vertical_threat_blocking(self):
        """Test AI blocks vertical four-in-a-row threats."""
        setup_moves = [
            (2, 4, -1),  # Opponent stones vertically
            (3, 4, -1),
            (4, 4, -1),
            (5, 4, -1),
            (2, 2, 1),   # AI stones elsewhere
            (3, 3, 1),
        ]
        
        game = self._create_game_scenario("Vertical Threat", setup_moves, current_player=1)
        
        # AI must block at (1,4) or (6,4)
        expected_blocks = [(1, 4), (6, 4)]
        is_correct, move = self._test_ai_response(game, expected_blocks, "Block vertical threat")
        
        self.assertTrue(is_correct, f"AI should block vertical threat, chose {move}")
    
    def test_diagonal_threat_blocking(self):
        """Test AI blocks diagonal four-in-a-row threats."""
        setup_moves = [
            (2, 2, -1),  # Opponent diagonal
            (3, 3, -1),
            (4, 4, -1),
            (5, 5, -1),
            (2, 6, 1),   # AI stones elsewhere
            (3, 7, 1),
        ]
        
        game = self._create_game_scenario("Diagonal Threat", setup_moves, current_player=1)
        
        # AI must block at (1,1) or (6,6)
        expected_blocks = [(1, 1), (6, 6)]
        is_correct, move = self._test_ai_response(game, expected_blocks, "Block diagonal threat")
        
        self.assertTrue(is_correct, f"AI should block diagonal threat, chose {move}")
    
    def test_winning_vs_blocking_priority(self):
        """Test AI prioritizes winning when it can win immediately."""
        setup_moves = [
            # AI can win horizontally
            (3, 2, 1),
            (3, 3, 1),
            (3, 4, 1),
            (3, 5, 1),  # AI four in a row, can win at (3,1) or (3,6)
            
            # Opponent threat (but AI should win instead of blocking)
            (6, 2, -1),
            (6, 3, -1),
            (6, 4, -1),  # Only three in a row - not immediate threat
        ]
        
        game = self._create_game_scenario("Win vs Block", setup_moves, current_player=1)
        
        # AI should win at (3,1) or (3,6)
        expected_wins = [(3, 1), (3, 6)]
        is_correct, move = self._test_ai_response(game, expected_wins, "Win the game")
        
        self.assertTrue(is_correct, f"AI should win instead of block, chose {move}")
    
    def test_multiple_threats_handling(self):
        """Test AI handles multiple simultaneous threats."""
        setup_moves = [
            # Threat 1: Horizontal
            (2, 2, -1),
            (2, 3, -1),
            (2, 4, -1),
            (2, 5, -1),  # Blocks needed at (2,1) or (2,6)
            
            # Threat 2: Vertical  
            (4, 7, -1),
            (5, 7, -1),
            (6, 7, -1),
            (7, 7, -1),  # Block needed at (3,7)
            
            # Some AI stones
            (4, 2, 1),
            (5, 3, 1),
        ]
        
        game = self._create_game_scenario("Multiple Threats", setup_moves, current_player=1)
        
        # AI should block one of the threats
        expected_blocks = [(2, 1), (2, 6), (3, 7)]
        is_correct, move = self._test_ai_response(game, expected_blocks, "Block one of multiple threats")
        
        self.assertTrue(is_correct, f"AI should block a threat, chose {move}")
    
    def test_capture_opportunity_vs_blocking(self):
        """Test AI behavior when it can capture or block."""
        setup_moves = [
            # Capture opportunity: AI stone, two opponent stones, can capture at end
            (3, 1, 1),   # AI stone
            (3, 2, -1),  # Opponent stones
            (3, 3, -1),
            # AI can capture by playing (3, 4)
            
            # Blocking opportunity: opponent threat
            (6, 2, -1),
            (6, 3, -1),
            (6, 4, -1),
            (6, 5, -1),  # Block needed at (6,1) or (6,6)
        ]
        
        game = self._create_game_scenario("Capture vs Block", setup_moves, current_player=1)
        
        # Should prioritize blocking over capture
        expected_moves = [(6, 1), (6, 6)]  # Blocking moves
        is_correct, move = self._test_ai_response(game, expected_moves, "Block threat (priority over capture)")
        
        self.assertTrue(is_correct, f"AI should block critical threat over capture, chose {move}")
    
    def test_endgame_blocking_scenario(self):
        """Test AI blocking in a complex endgame position."""
        setup_moves = [
            # Complex position with scattered stones
            (1, 1, 1), (1, 3, 1), (2, 2, 1),  # AI stones
            (2, 4, -1), (3, 5, -1), (4, 1, -1),  # Opponent stones
            
            # Critical threat: opponent about to win
            (5, 2, -1),
            (5, 3, -1),
            (5, 4, -1),
            (5, 5, -1),  # Must block at (5,1) or (5,6)
        ]
        
        game = self._create_game_scenario("Complex Endgame", setup_moves, current_player=1)
        
        expected_blocks = [(5, 1), (5, 6)]
        is_correct, move = self._test_ai_response(game, expected_blocks, "Block in complex position")
        
        self.assertTrue(is_correct, f"AI should find critical block in complex position, chose {move}")
    
    def test_ai_consistency(self):
        """Test AI consistency across multiple runs of the same position."""
        setup_moves = [
            (4, 2, -1),
            (4, 3, -1), 
            (4, 4, -1),
            (4, 5, -1),  # Clear threat requiring block
        ]
        
        game = self._create_game_scenario("Consistency Test", setup_moves, current_player=1)
        expected_blocks = [(4, 1), (4, 6)]
        
        # Test multiple times
        ai_player = MCTSPlayer(name="Test AI", player_id=1, max_iterations=self.ai_iterations)
        
        results = []
        for i in range(5):
            move = ai_player.get_move(game.clone())
            is_correct = move in expected_blocks
            results.append(is_correct)
            print(f"Run {i+1}: {move} - {'✓' if is_correct else '✗'}")
        
        success_rate = sum(results) / len(results)
        print(f"Success rate: {success_rate*100:.1f}%")
        
        # Should be consistent (at least 80% success)
        self.assertGreaterEqual(success_rate, 0.8, 
                              f"AI should be consistent, got {success_rate*100:.1f}% success rate")
    
    def test_performance_benchmark(self):
        """Benchmark AI response time for blocking decisions."""
        import time
        
        setup_moves = [(4, 2, -1), (4, 3, -1), (4, 4, -1), (4, 5, -1)]
        game = self._create_game_scenario("Performance Test", setup_moves, current_player=1)
        
        ai_player = MCTSPlayer(name="Test AI", player_id=1, max_iterations=self.ai_iterations)
        
        start_time = time.time()
        move = ai_player.get_move(game)
        end_time = time.time()
        
        response_time = end_time - start_time
        expected_blocks = [(4, 1), (4, 6)]
        is_correct = move in expected_blocks
        
        print(f"Response time: {response_time:.3f}s")
        print(f"Move: {move} - {'✓' if is_correct else '✗'}")
        
        # Should respond within reasonable time (< 5 seconds for 100 iterations)
        self.assertLess(response_time, 5.0, 
                       f"AI should respond quickly, took {response_time:.3f}s")
        self.assertTrue(is_correct, f"AI should find correct move, chose {move}")


if __name__ == '__main__':
    unittest.main(verbosity=2)