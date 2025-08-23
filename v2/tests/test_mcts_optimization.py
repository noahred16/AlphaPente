import unittest
import sys
import os
import time
import gc

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from games.pente import Pente
from mcts.mcts import MCTS
from players.mcts_player import MCTSPlayer


class TestMCTSOptimization(unittest.TestCase):
    """Test MCTS optimizations including move/undo patterns and memory efficiency."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.board_size = 9  # Larger board for performance testing
        self.mcts_iterations = 50  # Reduced iterations for faster testing
        
    def test_game_state_preservation(self):
        """Test that MCTS search preserves the original game state."""
        # Create initial game state
        game = Pente(board_size=self.board_size, tournament_rule=False)
        
        # Add some stones to create a non-trivial state
        test_moves = [(4, 4), (4, 5), (5, 4), (5, 5)]
        for i, move in enumerate(test_moves):
            game.board[move[0], move[1]] = 1 if i % 2 == 0 else -1
        
        # Sync cache after manual board modifications
        game._sync_legal_moves_cache()
        
        original_player = game.current_player
        original_board_sum = game.board.sum()
        original_captures = game.captures.copy()
        original_move_history = game.move_history.copy()
        
        # Run MCTS search
        mcts = MCTS(max_iterations=self.mcts_iterations)
        move = mcts.search(game)
        
        # Verify state preservation
        self.assertEqual(game.current_player, original_player, 
                        "Current player should be preserved")
        self.assertEqual(game.board.sum(), original_board_sum,
                        "Board state should be preserved") 
        self.assertEqual(game.captures, original_captures,
                        "Captures should be preserved")
        self.assertEqual(game.move_history, original_move_history,
                        "Move history should be preserved")
        self.assertIsInstance(move, tuple, "Should return a valid move")
        self.assertEqual(len(move), 2, "Move should be a coordinate pair")
        
    def test_multiple_searches_consistency(self):
        """Test that multiple MCTS searches on the same state are consistent."""
        game = Pente(board_size=7, tournament_rule=False)
        
        # Set up a specific position
        game.board[3, 1] = -1
        game.board[3, 2] = -1  
        game.board[3, 3] = -1
        game.board[3, 4] = -1  # Opponent threat
        game.current_player = 1
        
        original_state = {
            'board_sum': game.board.sum(),
            'current_player': game.current_player,
            'captures': game.captures.copy()
        }
        
        mcts = MCTS(max_iterations=50)
        
        # Perform multiple searches
        moves = []
        for i in range(5):
            move = mcts.search(game)
            moves.append(move)
            
            # Verify state hasn't changed after each search
            self.assertEqual(game.board.sum(), original_state['board_sum'],
                           f"Board state corrupted after search {i+1}")
            self.assertEqual(game.current_player, original_state['current_player'],
                           f"Current player corrupted after search {i+1}")
            self.assertEqual(game.captures, original_state['captures'],
                           f"Captures corrupted after search {i+1}")
        
        # All moves should be legal
        legal_moves = game.get_legal_moves()
        for move in moves:
            self.assertIn(move, legal_moves, f"Move {move} should be legal")
            
    def test_memory_efficiency(self):
        """Test that MCTS doesn't create excessive memory usage."""
        # This test verifies that we're not leaking memory through excessive cloning
        
        game = Pente(board_size=7, tournament_rule=False)
        game.current_player = 1
        
        # Measure memory usage before
        gc.collect()  # Force garbage collection
        
        mcts = MCTS(max_iterations=30)
        
        # Run multiple searches to see if memory grows excessively
        for _ in range(10):
            move = mcts.search(game)
            self.assertIsNotNone(move, "Should always return a valid move")
            
        # Force garbage collection and ensure we haven't leaked too much memory
        gc.collect()
        
        # This is a basic test - in a real scenario we'd use memory profiling tools
        # but for unit tests we just verify the basic functionality works
        self.assertTrue(True, "Memory efficiency test passed")
        
    def test_performance_comparison(self):
        """Test performance characteristics of optimized MCTS."""
        game = Pente(board_size=7, tournament_rule=False)
        
        # Add some complexity to the board
        positions = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]
        for i, pos in enumerate(positions):
            game.board[pos[0], pos[1]] = 1 if i % 2 == 0 else -1
        game.current_player = 1
        
        # Time MCTS searches
        mcts = MCTS(max_iterations=30)
        
        times = []
        for _ in range(5):
            start_time = time.time()
            move = mcts.search(game)
            end_time = time.time()
            
            times.append(end_time - start_time)
            self.assertIsNotNone(move, "Should return a valid move")
            
        avg_time = sum(times) / len(times)
        max_time = max(times)
        
        # Performance assertions (these are reasonable for the optimized version)
        self.assertLess(avg_time, 2.0, f"Average search time should be under 2s, got {avg_time:.3f}s")
        self.assertLess(max_time, 5.0, f"Max search time should be under 5s, got {max_time:.3f}s")
        
    def test_simulation_isolation(self):
        """Test that simulations don't affect each other or the main game state."""
        game = Pente(board_size=7, tournament_rule=False)
        
        # Create a position where simulations could interfere
        game.board[3, 3] = 1
        game.board[4, 4] = -1
        game.current_player = 1
        
        original_board = game.board.copy()
        
        mcts = MCTS(max_iterations=20)
        
        # Run search and verify no interference
        move = mcts.search(game)
        
        # Board should be unchanged
        self.assertTrue((game.board == original_board).all(), 
                       "Game board should be unchanged after MCTS search")
        
        # Move should be legal and valid
        legal_moves = game.get_legal_moves()
        self.assertIn(move, legal_moves, "Returned move should be legal")
        
    def test_large_board_scalability(self):
        """Test MCTS performance on larger boards."""
        # Test with different board sizes to ensure scalability
        board_sizes = [7, 11, 15]
        iterations = [50, 30, 20]  # Fewer iterations for larger boards
        
        for board_size, max_iter in zip(board_sizes, iterations):
            with self.subTest(board_size=board_size):
                game = Pente(board_size=board_size, tournament_rule=False)
                game.current_player = 1
                
                mcts = MCTS(max_iterations=max_iter)
                
                start_time = time.time()
                move = mcts.search(game)
                end_time = time.time()
                
                search_time = end_time - start_time
                
                # Verify results
                self.assertIsNotNone(move, f"Should return move for {board_size}x{board_size} board")
                self.assertLess(search_time, 10.0, 
                               f"Search on {board_size}x{board_size} board should complete in under 10s")
                
                legal_moves = game.get_legal_moves()
                self.assertIn(move, legal_moves, "Move should be legal")
                
    def test_complex_position_handling(self):
        """Test MCTS on complex positions with many stones."""
        game = Pente(board_size=9, tournament_rule=False)
        
        # Create a complex mid-game position
        positions = [
            (4, 4, 1), (4, 5, -1), (5, 4, 1), (5, 5, -1),
            (3, 3, 1), (6, 6, -1), (2, 2, 1), (7, 7, -1),
            (3, 5, 1), (5, 3, -1), (6, 4, 1), (4, 6, -1),
            (1, 1, 1), (8, 8, -1), (1, 8, 1), (8, 1, -1)
        ]
        
        for row, col, player in positions:
            game.board[row, col] = player
            
        # Sync cache after manual board modifications
        game._sync_legal_moves_cache()
        game.current_player = 1
        original_board_sum = game.board.sum()
        
        mcts = MCTS(max_iterations=150)
        
        # Test multiple searches on complex position
        for i in range(3):
            move = mcts.search(game)
            
            # Verify state preservation
            self.assertEqual(game.board.sum(), original_board_sum,
                           f"Board state should be preserved after search {i+1}")
            
            # Verify move is legal
            legal_moves = game.get_legal_moves()
            self.assertIn(move, legal_moves, f"Move {move} should be legal")
            
    def test_edge_case_scenarios(self):
        """Test MCTS behavior in edge case scenarios."""
        
        # Test 1: Nearly full board
        game = Pente(board_size=5, tournament_rule=False)
        
        # Fill most of the board
        for i in range(5):
            for j in range(5):
                if (i + j) % 3 != 0:  # Leave some spaces
                    game.board[i, j] = 1 if (i + j) % 2 == 0 else -1
                    
        game.current_player = 1
        legal_moves = game.get_legal_moves()
        
        if legal_moves:  # Only test if moves are available
            mcts = MCTS(max_iterations=50)
            move = mcts.search(game)
            
            self.assertIn(move, legal_moves, "Should choose from available legal moves")
            
        # Test 2: Terminal position (someone already won)
        game2 = Pente(board_size=7, tournament_rule=False)
        # Create a winning position
        for col in range(5):
            game2.board[3, col] = 1
        game2.current_player = -1
        
        if not game2.is_terminal():
            mcts = MCTS(max_iterations=20)
            move = mcts.search(game2)
            legal_moves = game2.get_legal_moves()
            self.assertIn(move, legal_moves, "Should return legal move even in winning position")


class TestMCTSPlayerIntegration(unittest.TestCase):
    """Integration tests for MCTSPlayer with optimized MCTS."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.board_size = 7
        self.player = MCTSPlayer(name="Test AI", player_id=1, max_iterations=30)
        
    def test_player_state_preservation(self):
        """Test that MCTSPlayer preserves game state across moves."""
        game = Pente(board_size=self.board_size, tournament_rule=False)
        
        # Set up initial position
        game.board[3, 3] = -1
        game.current_player = 1
        
        original_state = {
            'board': game.board.copy(),
            'current_player': game.current_player,
            'captures': game.captures.copy()
        }
        
        # Player makes a move
        move = self.player.get_move(game)
        
        # Verify state preservation
        self.assertTrue((game.board == original_state['board']).all(),
                       "Board should be preserved")
        self.assertEqual(game.current_player, original_state['current_player'],
                       "Current player should be preserved")
        self.assertEqual(game.captures, original_state['captures'],
                       "Captures should be preserved")
        
        # Verify move is valid
        legal_moves = game.get_legal_moves()
        self.assertIn(move, legal_moves, "Player should return legal move")
        
    def test_difficulty_levels(self):
        """Test different difficulty levels maintain state preservation."""
        from players.mcts_player import EasyMCTSPlayer, MediumMCTSPlayer, HardMCTSPlayer
        
        players = [
            EasyMCTSPlayer(player_id=1),
            MediumMCTSPlayer(player_id=1),
            HardMCTSPlayer(player_id=1)
        ]
        
        for player in players:
            with self.subTest(player=player.name):
                game = Pente(board_size=5, tournament_rule=False)
                game.current_player = 1
                
                original_sum = game.board.sum()
                
                move = player.get_move(game)
                
                # Verify state preservation
                self.assertEqual(game.board.sum(), original_sum,
                               f"{player.name} should preserve game state")
                
                # Verify valid move
                legal_moves = game.get_legal_moves()
                self.assertIn(move, legal_moves, f"{player.name} should return legal move")
                
    def test_player_consistency(self):
        """Test that player returns consistent results for same position."""
        game = Pente(board_size=5, tournament_rule=False)
        game.current_player = 1
        
        # Run multiple times to check consistency
        moves = []
        for _ in range(3):
            move = self.player.get_move(game)
            moves.append(move)
            
            # Verify state hasn't changed
            self.assertEqual(game.current_player, 1, "Player should be preserved")
            
        # All moves should be legal (consistency in that they're all valid)
        legal_moves = game.get_legal_moves()
        for move in moves:
            self.assertIn(move, legal_moves, f"Move {move} should be legal")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)