import unittest
import sys
import os
from typing import List, Tuple
import time

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from games.pente import Pente
from players.mcts_player import MCTSPlayer


class TestMCTSAnalysis(unittest.TestCase):
    """Comprehensive analysis of MCTS performance with 50 simulations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.board_size = 7
        self.mcts_iterations = 100
        self.test_scenarios = []
        
    def _create_test_scenario(self, name: str, opponent_stones: List[Tuple[int, int]], 
                            expected_blocks: List[Tuple[int, int]], description: str):
        """Create a standardized test scenario."""
        return {
            'name': name,
            'opponent_stones': opponent_stones,
            'expected_blocks': expected_blocks,
            'description': description
        }
    
    def _setup_scenario(self, scenario: dict) -> Tuple[Pente, MCTSPlayer]:
        """Set up game state for a specific scenario."""
        game = Pente(board_size=self.board_size, tournament_rule=False)
        player = MCTSPlayer(name="MCTS Test", player_id=1, max_iterations=self.mcts_iterations)
        
        # Place opponent stones
        for row, col in scenario['opponent_stones']:
            game.board[row, col] = -1
        
        # Synchronize legal moves cache after manual board setup
        game._sync_legal_moves_cache()
            
        game.current_player = 1
        return game, player
    
    def test_all_blocking_scenarios_analysis(self):
        """Comprehensive test of all blocking scenarios with performance analysis."""
        
        # Define test scenarios
        scenarios = [
            self._create_test_scenario(
                "Horizontal Four", 
                [(3, 1), (3, 2), (3, 3), (3, 4)],
                [(3, 0), (3, 5)],
                "Four stones in a row horizontally"
            ),
            self._create_test_scenario(
                "Vertical Four", 
                [(1, 3), (2, 3), (3, 3), (4, 3)],
                [(0, 3), (5, 3), (6, 3)],
                "Four stones in a row vertically"
            ),
            self._create_test_scenario(
                "Diagonal Four (\\)", 
                [(1, 1), (2, 2), (3, 3), (4, 4)],
                [(0, 0), (5, 5), (6, 6)],
                "Four stones in diagonal line (top-left to bottom-right)"
            ),
            self._create_test_scenario(
                "Diagonal Four (/)", 
                [(4, 1), (3, 2), (2, 3), (1, 4)],
                [(5, 0), (0, 5), (6, 0)],
                "Four stones in diagonal line (bottom-left to top-right)"
            ),
            self._create_test_scenario(
                "Edge Four", 
                [(0, 1), (0, 2), (0, 3), (0, 4)],
                [(0, 0), (0, 5)],
                "Four stones along the top edge"
            ),
            self._create_test_scenario(
                "Corner Four", 
                [(6, 2), (6, 3), (6, 4), (6, 5)],
                [(6, 1), (6, 6)],
                "Four stones near the bottom edge"
            )
        ]
        
        print(f"\n{'='*80}")
        print(f"MCTS BLOCKING ANALYSIS - {self.mcts_iterations} Simulations")
        print(f"{'='*80}")
        
        results = []
        total_tests = 0
        successful_blocks = 0
        
        for scenario in scenarios:
            print(f"\nTesting: {scenario['name']}")
            print(f"Description: {scenario['description']}")
            print(f"Opponent stones: {scenario['opponent_stones']}")
            print(f"Expected blocking positions: {scenario['expected_blocks']}")
            
            # Run multiple trials for statistical analysis
            trial_results = []
            times = []
            
            for trial in range(5):
                game, player = self._setup_scenario(scenario)
                
                start_time = time.time()
                move = player.get_move(game)
                end_time = time.time()
                
                times.append(end_time - start_time)
                is_blocking = move in scenario['expected_blocks']
                trial_results.append({
                    'move': move,
                    'is_blocking': is_blocking,
                    'time': end_time - start_time
                })
                
                total_tests += 1
                if is_blocking:
                    successful_blocks += 1
            
            # Calculate statistics
            blocking_rate = sum(1 for r in trial_results if r['is_blocking']) / len(trial_results)
            avg_time = sum(times) / len(times)
            
            print(f"Results:")
            for i, result in enumerate(trial_results):
                status = "✓ BLOCKED" if result['is_blocking'] else "✗ MISSED"
                print(f"  Trial {i+1}: {result['move']} - {status} ({result['time']:.3f}s)")
            
            print(f"Blocking Rate: {blocking_rate*100:.1f}% ({sum(1 for r in trial_results if r['is_blocking'])}/{len(trial_results)})")
            print(f"Average Time: {avg_time:.3f}s")
            
            results.append({
                'scenario': scenario['name'],
                'blocking_rate': blocking_rate,
                'avg_time': avg_time,
                'trials': trial_results
            })
            
            # Assert that MCTS blocks correctly in majority of trials
            self.assertGreaterEqual(blocking_rate, 0.6, 
                                  f"MCTS should block {scenario['name']} in at least 60% of trials, got {blocking_rate*100:.1f}%")
        
        print(f"\n{'='*80}")
        print(f"OVERALL RESULTS")
        print(f"{'='*80}")
        print(f"Total Tests: {total_tests}")
        print(f"Successful Blocks: {successful_blocks}")
        print(f"Overall Success Rate: {successful_blocks/total_tests*100:.1f}%")
        
        # Summary by scenario type
        print(f"\nSummary by Scenario:")
        for result in results:
            print(f"  {result['scenario']:<20} {result['blocking_rate']*100:>5.1f}%   {result['avg_time']:>6.3f}s")
        
        overall_success_rate = successful_blocks / total_tests
        self.assertGreaterEqual(overall_success_rate, 0.7, 
                              f"Overall MCTS blocking success rate should be at least 70%, got {overall_success_rate*100:.1f}%")
        
        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*80}")
    
    def test_winning_vs_blocking_priority(self):
        """Test MCTS behavior with critical moves (both get same heuristic score)."""
        print(f"\n{'='*60}")
        print("TESTING CRITICAL MOVE RECOGNITION")
        print(f"{'='*60}")
        
        game = Pente(board_size=self.board_size, tournament_rule=False)
        player = MCTSPlayer(name="MCTS Test", player_id=1, max_iterations=self.mcts_iterations)
        
        # Set up winning opportunity for MCTS
        mcts_stones = [(2, 1), (2, 2), (2, 3), (2, 4)]  # Can win at (2,5) or (2,0)
        for row, col in mcts_stones:
            game.board[row, col] = 1
            
        # Set up blocking opportunity - opponent threat (different row)
        opponent_stones = [(5, 1), (5, 2), (5, 3), (5, 4)]  # Threat at (5,5) or (5,0)
        for row, col in opponent_stones:
            game.board[row, col] = -1
        
        # Synchronize legal moves cache after manual board setup
        game._sync_legal_moves_cache()
            
        game.current_player = 1
        
        # Test multiple times
        winning_moves = [(2, 5), (2, 0)]
        blocking_moves = [(5, 5), (5, 0)]
        critical_moves = winning_moves + blocking_moves
        
        results = []
        for trial in range(5):
            move = player.get_move(game)
            is_winning = move in winning_moves
            is_blocking = move in blocking_moves
            is_critical = move in critical_moves
            results.append({
                'move': move,
                'is_winning': is_winning,
                'is_blocking': is_blocking,
                'is_critical': is_critical
            })
            
            print(f"Trial {trial+1}: {move} - {'WIN' if is_winning else 'BLOCK' if is_blocking else 'OTHER'}")
        
        # MCTS should recognize critical moves (either winning or blocking)
        critical_count = sum(1 for r in results if r['is_critical'])
        critical_rate = critical_count / len(results)
        
        print(f"\nCritical moves chosen: {critical_count}/{len(results)} ({critical_rate*100:.1f}%)")
        print("Note: Both winning and blocking moves have 'critical' priority in the heuristic system")
        
        # Accept either winning or blocking as both are critical
        self.assertGreaterEqual(critical_rate, 0.8,
                              f"MCTS should choose critical moves (win or block) in at least 80% of cases, got {critical_rate*100:.1f}%")
    
    def test_heuristic_effectiveness(self):
        """Test the effectiveness of heuristics in guiding MCTS decisions."""
        print(f"\n{'='*60}")
        print("TESTING HEURISTIC EFFECTIVENESS")
        print(f"{'='*60}")
        
        # Create a position with multiple tactical elements
        game = Pente(board_size=self.board_size, tournament_rule=False)
        player = MCTSPlayer(name="MCTS Test", player_id=1, max_iterations=self.mcts_iterations)
        
        # Critical threat - opponent about to win (only one blocking position)
        critical_threat = [(3, 0), (3, 1), (3, 2), (3, 3)]  # Block at (3,4) only
        for row, col in critical_threat:
            game.board[row, col] = -1
            
        # High priority - MCTS can create open three
        mcts_stones = [(5, 2), (5, 3)]  # Can create open three at (5,1) or (5,4)
        for row, col in mcts_stones:
            game.board[row, col] = 1
            
        # Medium priority - capture opportunity
        game.board[1, 1] = 1  # MCTS stone
        game.board[1, 2] = -1  # Opponent stones
        game.board[1, 3] = -1
        # Can capture at (1,4)
        
        # Synchronize legal moves cache after manual board setup
        game._sync_legal_moves_cache()
        
        game.current_player = 1

        # print game board
        print("Current Game Board:")
        print(game.board)
        
        # Test move selection
        results = []
        for trial in range(10):
            move = player.get_move(game)
            
            # Categorize the move
            if move == (3, 4):  # Only one blocking position
                category = "CRITICAL_BLOCK"
                priority = 4
            elif move in [(5, 1), (5, 4)]:
                category = "OPEN_THREE"
                priority = 3
            elif move == (1, 4):
                category = "CAPTURE"
                priority = 2
            else:
                category = "OTHER"
                priority = 1
                
            results.append({
                'move': move,
                'category': category,
                'priority': priority
            })
        
        # Analyze results
        category_counts = {}
        for result in results:
            category = result['category']
            category_counts[category] = category_counts.get(category, 0) + 1
            # print each move with its category
            print(f"Move: {result['move']} - Category: {category}")
        
        print("Move Distribution:")
        for category, count in sorted(category_counts.items(), reverse=True):
            print(f"  {category}: {count}/10 ({count*10:.1f}%)")
        
        # MCTS should prioritize critical moves
        critical_moves = sum(1 for r in results if r['priority'] >= 3)
        critical_rate = critical_moves / len(results)
        
        print(f"\nHigh Priority Moves (Critical/Open Three): {critical_moves}/10 ({critical_rate*100:.1f}%)")
        
        self.assertGreaterEqual(critical_rate, 0.7,
                              f"MCTS should make high-priority moves at least 70% of the time, got {critical_rate*100:.1f}%")


if __name__ == '__main__':
    unittest.main(verbosity=2)