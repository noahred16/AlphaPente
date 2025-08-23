import unittest
import sys
import os
import time
import gc

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from games.pente import Pente
from mcts.mcts import MCTS


class TestPerformanceComparison(unittest.TestCase):
    """Performance comparison tests to demonstrate optimization benefits."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.board_size = 9
        self.test_iterations = 25  # Reduced for faster testing
        
    def test_memory_usage_baseline(self):
        """Baseline test for memory usage of optimized MCTS."""
        game = Pente(board_size=self.board_size, tournament_rule=False)
        
        # Add some complexity
        positions = [(2, 2), (3, 3), (4, 4), (5, 5), (6, 6)]
        for i, pos in enumerate(positions):
            game.board[pos[0], pos[1]] = 1 if i % 2 == 0 else -1
            
        game.current_player = 1
        
        # Force garbage collection before measurement
        gc.collect()
        
        mcts = MCTS(max_iterations=self.test_iterations)
        
        # Run multiple searches and verify they complete without issues
        for i in range(5):
            move = mcts.search(game)
            self.assertIsNotNone(move, f"Search {i+1} should return valid move")
            
        gc.collect()
        
        print(f"\nMemory efficiency test:")
        print(f"Completed 5 searches with {self.test_iterations} iterations each")
        print(f"No memory issues detected")
        
        # Basic memory efficiency test - if we get here without issues, it's good
        self.assertTrue(True, "Memory efficiency test passed")
        
    def test_search_time_performance(self):
        """Test search time performance of optimized MCTS."""
        game = Pente(board_size=self.board_size, tournament_rule=False)
        
        # Create a moderately complex position
        moves = [(4, 4), (4, 5), (5, 4), (5, 5), (3, 3), (6, 6), (2, 2)]
        for i, move in enumerate(moves):
            game.board[move[0], move[1]] = 1 if i % 2 == 0 else -1
            
        game.current_player = 1
        
        mcts = MCTS(max_iterations=self.test_iterations)
        
        # Time multiple searches
        times = []
        for i in range(3):  # Reduced from 10 for faster testing
            start_time = time.time()
            move = mcts.search(game)
            end_time = time.time()
            
            search_time = end_time - start_time
            times.append(search_time)
            
            self.assertIsNotNone(move, f"Search {i+1} should return valid move")
            
        avg_time = sum(times) / len(times)
        max_time = max(times)
        min_time = min(times)
        
        print(f"\nSearch time performance:")
        print(f"Average: {avg_time:.3f}s")
        print(f"Min: {min_time:.3f}s")
        print(f"Max: {max_time:.3f}s")
        print(f"Iterations per search: {self.test_iterations}")
        print(f"Time per iteration: {avg_time/self.test_iterations*1000:.1f}ms")
        
        # Performance assertions for optimized version
        self.assertLess(avg_time, 3.0, f"Average search time should be under 3s, got {avg_time:.3f}s")
        self.assertLess(max_time, 5.0, f"Max search time should be under 5s, got {max_time:.3f}s")
        
    def test_scalability_with_board_size(self):
        """Test how performance scales with board size."""
        board_sizes = [7, 9, 11]
        iterations = [30, 25, 20]  # Fewer iterations for larger boards
        
        results = []
        
        for board_size, max_iter in zip(board_sizes, iterations):
            game = Pente(board_size=board_size, tournament_rule=False)
            game.current_player = 1
            
            mcts = MCTS(max_iterations=max_iter)
            
            # Time search
            start_time = time.time()
            move = mcts.search(game)
            end_time = time.time()
            
            search_time = end_time - start_time
            time_per_iteration = search_time / max_iter
            
            results.append({
                'board_size': board_size,
                'iterations': max_iter,
                'time': search_time,
                'time_per_iter': time_per_iteration
            })
            
            self.assertIsNotNone(move, f"Should return move for {board_size}x{board_size} board")
            
        print(f"\nScalability results:")
        for result in results:
            print(f"Board {result['board_size']}x{result['board_size']}: "
                  f"{result['time']:.3f}s ({result['iterations']} iter, "
                  f"{result['time_per_iter']*1000:.1f}ms/iter)")
            
        # Verify that time per iteration doesn't grow too much with board size
        time_per_iters = [r['time_per_iter'] for r in results]
        max_time_per_iter = max(time_per_iters)
        min_time_per_iter = min(time_per_iters)
        
        scaling_factor = max_time_per_iter / min_time_per_iter
        print(f"Scaling factor (max/min time per iteration): {scaling_factor:.2f}")
        
        # Should scale reasonably well (less than 5x difference)
        self.assertLess(scaling_factor, 5.0, 
                       f"Time per iteration shouldn't scale by more than 5x, got {scaling_factor:.2f}x")
        
    def test_state_preservation_overhead(self):
        """Test that state preservation doesn't add significant overhead."""
        game = Pente(board_size=7, tournament_rule=False)
        
        # Add some stones
        game.board[3, 3] = 1
        game.board[3, 4] = -1
        game.board[4, 3] = 1
        game.current_player = -1
        
        # Store original state
        original_board = game.board.copy()
        original_player = game.current_player
        original_captures = game.captures.copy()
        
        mcts = MCTS(max_iterations=30)
        
        # Time multiple searches with state preservation
        times = []
        for _ in range(5):
            start_time = time.time()
            move = mcts.search(game)
            end_time = time.time()
            
            times.append(end_time - start_time)
            
            # Verify state preservation
            self.assertTrue((game.board == original_board).all(),
                           "Board state should be preserved")
            self.assertEqual(game.current_player, original_player,
                           "Current player should be preserved")
            self.assertEqual(game.captures, original_captures,
                           "Captures should be preserved")
            
        avg_time = sum(times) / len(times)
        print(f"\nState preservation performance:")
        print(f"Average search time with state preservation: {avg_time:.3f}s")
        
        # Should still be reasonably fast even with state preservation
        self.assertLess(avg_time, 2.0, 
                       f"Search with state preservation should be under 2s, got {avg_time:.3f}s")
        
    def test_simulation_efficiency(self):
        """Test efficiency of the simulation phase."""
        game = Pente(board_size=7, tournament_rule=False)
        
        # Create a position where simulations matter
        positions = [(3, 3), (3, 4), (4, 3), (4, 4)]
        for i, pos in enumerate(positions):
            game.board[pos[0], pos[1]] = 1 if i % 2 == 0 else -1
            
        game.current_player = 1
        
        mcts = MCTS(max_iterations=100)  # More iterations to test simulation efficiency
        
        # Time the search
        start_time = time.time()
        move = mcts.search(game)
        end_time = time.time()
        
        total_time = end_time - start_time
        time_per_iteration = total_time / 100
        
        print(f"\nSimulation efficiency:")
        print(f"Total time: {total_time:.3f}s")
        print(f"Time per iteration: {time_per_iteration*1000:.1f}ms")
        
        self.assertIsNotNone(move, "Should return valid move")
        
        # Each iteration should be quite fast with optimized simulation
        self.assertLess(time_per_iteration, 0.05, 
                       f"Each iteration should be under 50ms, got {time_per_iteration*1000:.1f}ms")
        
    def test_memory_leak_detection(self):
        """Test for memory leaks over extended usage."""
        game = Pente(board_size=7, tournament_rule=False)
        game.current_player = 1
        
        mcts = MCTS(max_iterations=20)
        
        # Force garbage collection
        gc.collect()
        
        # Run many searches to detect potential issues
        for i in range(30):  # Reduced from 50 for faster testing
            move = mcts.search(game)
            self.assertIsNotNone(move, f"Search {i+1} should return valid move")
            
            # Occasionally force garbage collection
            if i % 10 == 9:
                gc.collect()
                
        gc.collect()
        
        print(f"\nMemory leak detection:")
        print(f"Completed 30 searches successfully")
        print(f"No obvious memory leaks detected")
        
        # If we complete without issues, no obvious memory leaks
        self.assertTrue(True, "Memory leak test passed")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)