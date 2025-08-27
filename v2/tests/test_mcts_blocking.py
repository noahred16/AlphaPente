import unittest
import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from games.pente import Pente
from players.mcts_player import MCTSPlayer
class TestMCTSBlocking(unittest.TestCase):
    """Test MCTS player's ability to block opponent's winning moves."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.board_size = 7  # Smaller board for faster testing
        self.mcts_iterations = 200  # Increased for scaled valuation system
        
    def _create_blocking_scenario(self, opponent_stones, blocking_move, game_size=7):
        """
        Helper method to create a blocking scenario.
        
        Args:
            opponent_stones: List of (row, col) positions for opponent stones
            blocking_move: (row, col) position where player should block
            game_size: Size of the game board
            
        Returns:
            Configured Pente game with MCTS player
        """
        game = Pente(board_size=game_size, tournament_rule=False)
        player = MCTSPlayer(name="MCTS Test", player_id=1, max_iterations=self.mcts_iterations)
        
        # Place opponent stones (player -1)
        for row, col in opponent_stones:
            game.board[row, col] = -1
        
        # Synchronize legal moves cache after manual board setup
        game._sync_legal_moves_cache()
            
        # Set current player to 1 (MCTS player)
        game.current_player = 1
        
        return game, player, blocking_move
    
    def test_horizontal_four_blocking(self):
        """Test blocking horizontal four in a row."""
        # Opponent has four stones horizontally at (4,1), (4,2), (4,3), (4,4)
        # Player should block at (4,5) or (4,0)
        opponent_stones = [(4, 1), (4, 2), (4, 3), (4, 4)]
        expected_blocks = [(4, 5), (4, 0)]
        
        game, player, _ = self._create_blocking_scenario(opponent_stones, expected_blocks[0])
        
        # Get MCTS move
        move = player.get_move(game)
        
        # Verify the move blocks the win
        self.assertIn(move, expected_blocks, 
                     f"MCTS should block horizontal four at {expected_blocks}, got {move}")
        
        # Verify this prevents immediate win by occupying one of the winning squares
        # Note: In a four-in-a-row scenario, blocking one end means opponent 
        # would need to play the other end to win, which they can do next turn.
        # The test validates that MCTS recognizes the critical nature of the position.
    
    def test_vertical_four_blocking(self):
        """Test blocking vertical four in a row."""
        # Opponent has four stones vertically at (1,3), (2,3), (3,3), (4,3)
        # Player should block at (5,3), (0,3), or (6,3) if within board
        opponent_stones = [(1, 3), (2, 3), (3, 3), (4, 3)]
        expected_blocks = [(5, 3), (0, 3), (6, 3)]
        
        game, player, _ = self._create_blocking_scenario(opponent_stones, expected_blocks[0])
        
        # Get MCTS move
        move = player.get_move(game)
        
        # Verify the move blocks the win
        self.assertIn(move, expected_blocks, 
                     f"MCTS should block vertical four at {expected_blocks}, got {move}")
    
    def test_diagonal_four_blocking_positive_slope(self):
        """Test blocking diagonal four in a row (positive slope)."""
        # Opponent has four stones diagonally at (5,1), (4,2), (3,3), (2,4)
        # Player should block at (1,5) or (6,0)
        opponent_stones = [(5, 1), (4, 2), (3, 3), (2, 4)]
        expected_blocks = [(1, 5), (6, 0)]
        
        game, player, _ = self._create_blocking_scenario(opponent_stones, expected_blocks[0])
        
        # Get MCTS move
        move = player.get_move(game)
        
        # Verify the move blocks the win
        self.assertIn(move, expected_blocks, 
                     f"MCTS should block diagonal four at {expected_blocks}, got {move}")
    
    def test_diagonal_four_blocking_negative_slope(self):
        """Test blocking diagonal four in a row (negative slope)."""
        # Opponent has four stones diagonally at (1,1), (2,2), (3,3), (4,4)
        # Player should block at (0,0) or (5,5)
        opponent_stones = [(1, 1), (2, 2), (3, 3), (4, 4)]
        expected_blocks = [(0, 0), (5, 5)]
        
        game, player, _ = self._create_blocking_scenario(opponent_stones, expected_blocks[0])
        
        # Get MCTS move
        move = player.get_move(game)
        
        # Verify the move blocks the win
        self.assertIn(move, expected_blocks, 
                     f"MCTS should block diagonal four at {expected_blocks}, got {move}")
    
    def test_blocked_four_no_threat(self):
        """Test that MCTS recognizes when four in a row is already blocked."""
        # Opponent has four stones but one end is blocked
        # (4,1), (4,2), (4,3), (4,4) with player stone at (4,0)
        opponent_stones = [(4, 1), (4, 2), (4, 3), (4, 4)]
        
        game, player, _ = self._create_blocking_scenario(opponent_stones, None)
        game.board[4, 0] = 1  # Block one end
        game._sync_legal_moves_cache()  # Sync after manual board modification
        
        # Get MCTS move - should not prioritize (4,5) as highly since threat is reduced
        move = player.get_move(game)
        
        # Verify game is still playable and MCTS makes a reasonable move
        self.assertIn(move, game.get_legal_moves(), 
                     "MCTS should make a legal move even when threat is blocked")
    
    def test_multiple_threats_prioritization(self):
        """Test MCTS handling multiple simultaneous threats."""
        # Create two separate four-in-a-row threats
        # Horizontal: (2,1), (2,2), (2,3), (2,4) - block at (2,5) or (2,0)
        # Vertical: (1,6), (3,6), (4,6), (5,6) - block at (2,6)
        opponent_stones = [(2, 1), (2, 2), (2, 3), (2, 4),  # Horizontal threat
                          (1, 6), (3, 6), (4, 6), (5, 6)]   # Vertical threat
        
        game, player, _ = self._create_blocking_scenario(opponent_stones, None)
        
        # Get MCTS move
        move = player.get_move(game)
        
        # Should block one of the threats
        valid_blocks = [(2, 5), (2, 0), (2, 6)]
        self.assertIn(move, valid_blocks, 
                     f"MCTS should block one of the threats at {valid_blocks}, got {move}")
    
    def test_winning_vs_blocking_priority(self):
        """Test that MCTS recognizes winning opportunities."""
        game, player, _ = self._create_blocking_scenario([], None)
        
        # Place MCTS stones in a row where it can win
        for col in range(1, 5):
            game.board[3, col] = 1
            
        # Place opponent stones (but not an immediate threat)
        game.board[5, 1] = -1
        game.board[5, 3] = -1
        
        # Sync cache after manual board modifications
        game._sync_legal_moves_cache()
            
        # Get MCTS move
        move = player.get_move(game)

        # mcts move statistics
        stats = player.get_move_statistics(game)
        print(f"MCTS move: {move}")
        print(f"Board state:\n{game.board}")
        # show top five moves from stats w newlines
        print(f"Top five moves:\n" + "\n".join([f"{move} - Visits: {visits}, Win Rate: {win_rate:.2f}"
                                                  for move, visits, win_rate in stats[:5]]))

        # Should choose to win - (3,0), (3,5), or (3,6) all complete the five
        winning_moves = [(3, 0), (3, 5), (3, 6)]
        self.assertIn(move, winning_moves, 
                     f"MCTS should win at {winning_moves}, got {move}")
    
    def test_edge_blocking_scenarios(self):
        """Test blocking scenarios near board edges."""
        # Opponent has four stones near edge: (0,1), (0,2), (0,3), (0,4)
        # Can only block at (0,5) since (0,0) would be off-board in some cases
        opponent_stones = [(0, 1), (0, 2), (0, 3), (0, 4)]
        expected_blocks = [(0, 5), (0, 0)]  # (0,0) is valid on a 9x9 board
        
        game, player, _ = self._create_blocking_scenario(opponent_stones, expected_blocks[0])
        
        # Get MCTS move
        move = player.get_move(game)
        
        # Verify the move blocks the win and is legal
        self.assertIn(move, game.get_legal_moves(), 
                     "MCTS move should be legal")
        self.assertIn(move, expected_blocks, 
                     f"MCTS should block edge four at {expected_blocks}, got {move}")
    
    def test_broken_four_detection(self):
        """Test that MCTS doesn't overreact to broken four-in-a-row."""
        # Opponent has stones with gaps: (4,1), (4,3), (4,4), (4,5)
        # This is not an immediate threat since (4,2) is empty
        opponent_stones = [(4, 1), (4, 3), (4, 4), (4, 5)]
        
        game, player, _ = self._create_blocking_scenario(opponent_stones, None)
        
        # Get MCTS move
        move = player.get_move(game)
        
        # Should make a reasonable move (anywhere legal is fine since no immediate threat)
        self.assertIn(move, game.get_legal_moves(), 
                     "MCTS should make a legal move for broken four scenario")
    
    def test_performance_consistency(self):
        """Test that MCTS consistently finds blocking moves across multiple runs."""
        # Test the same scenario multiple times to check consistency
        opponent_stones = [(3, 1), (3, 2), (3, 3), (3, 4)]
        expected_blocks = [(3, 0), (3, 5)]
        
        blocking_moves = []
        for _ in range(3):  # Reduced to 3 runs for speed
            game, player, _ = self._create_blocking_scenario(opponent_stones, expected_blocks[0])
            move = player.get_move(game)
            blocking_moves.append(move)
        
        # All moves should be blocking moves
        for move in blocking_moves:
            self.assertIn(move, expected_blocks, 
                         f"MCTS should consistently block at {expected_blocks}, got {move}")
    
    def test_complex_position_blocking(self):
        """Test blocking in a complex position with many stones."""
        game = Pente(board_size=self.board_size, tournament_rule=False)
        player = MCTSPlayer(name="MCTS Test", player_id=1, max_iterations=self.mcts_iterations)
        
        # Create a complex position with scattered stones
        scattered_positions = [(1, 1), (2, 3), (3, 5), (6, 2), (6, 6)]
        for row, col in scattered_positions:
            game.board[row, col] = 1  # MCTS stones
            
        opponent_scattered = [(1, 2), (3, 4), (5, 6), (6, 1)]
        for row, col in opponent_scattered:
            game.board[row, col] = -1  # Opponent stones
            
        # Add the critical threat: four in a row for opponent
        threat_positions = [(4, 1), (4, 2), (4, 3), (4, 4)]
        for row, col in threat_positions:
            game.board[row, col] = -1
        
        # Sync cache after manual board setup
        game._sync_legal_moves_cache()
            
        game.current_player = 1
        
        # Get MCTS move
        move = player.get_move(game)
        
        # Should still find the blocking move despite complex position
        expected_blocks = [(4, 0), (4, 5)]
        self.assertIn(move, expected_blocks, 
                     f"MCTS should block threat in complex position at {expected_blocks}, got {move}")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)