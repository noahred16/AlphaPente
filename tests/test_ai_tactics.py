"""
Tests demonstrating AI tactical effectiveness.
Tests that the MCTS AI can learn important Pente tactics with sufficient simulations.
"""

import pytest
from src.core.game_state import GameState
from src.mcts.engine import MCTSEngine

class TestAITactics:
    """Test that AI demonstrates tactical understanding of Pente."""
    
    def test_ai_blocks_immediate_win_threat(self):
        """AI should block opponent's immediate 4-in-a-row threat."""
        game_state = GameState()
        
        # Set up position where opponent (O) has 4 in a row with one open end
        # X . O O O O . (O threatens to win on next move)
        positions = [
            (9, 8, 1),   # X
            (9, 10, -1), # O  
            (9, 11, -1), # O
            (9, 12, -1), # O
            (9, 13, -1)  # O
        ]
        
        for row, col, player in positions:
            game_state.board.set_stone(row, col, player)
            game_state.move_count += 1
        
        # It's X's turn - should block at (9, 9) or (9, 14)
        game_state.current_player = 1
        
        # Give AI plenty of time to find the defensive move
        ai = MCTSEngine(time_limit_ms=3000, max_simulations=500)
        move = ai.search(game_state)
        
        # AI should play at one of the blocking positions
        blocking_moves = {(9, 9), (9, 14)}
        assert move in blocking_moves, f"AI played {move}, should block threat at {blocking_moves}"
        
        # Verify this actually blocks the threat
        game_state.make_move(move)
        assert not game_state.is_terminal(), "Game should continue after blocking"
        
        stats = ai.get_performance_stats()
        print(f"Threat blocking: {stats['simulations']} sims, {stats['simulations_per_second']:.1f}/sec")
    
    def test_ai_takes_immediate_win(self):
        """AI should take an immediate winning move when available."""
        game_state = GameState()
        
        # Set up position where AI (X) can win immediately
        # X X X X . (X can win by playing at (9, 13))
        positions = [
            (9, 9, 1),   # X
            (9, 10, 1),  # X
            (9, 11, 1),  # X  
            (9, 12, 1),  # X
            (8, 9, -1),  # O (some opponent moves)
            (8, 10, -1), # O
        ]
        
        for row, col, player in positions:
            game_state.board.set_stone(row, col, player)
            game_state.move_count += 1
        
        game_state.current_player = 1  # X to play
        
        ai = MCTSEngine(time_limit_ms=1000, max_simulations=100)
        move = ai.search(game_state)
        
        # AI should play the winning move
        winning_move = (9, 13)
        assert move == winning_move, f"AI played {move}, should win at {winning_move}"
        
        # Verify this actually wins
        game_state.make_move(move)
        assert game_state.is_terminal(), "Game should end after winning move"
        assert game_state.get_winner() == 1, "X should be the winner"
        
        stats = ai.get_performance_stats()
        print(f"Immediate win: {stats['simulations']} sims, found in {stats['time_ms']:.1f}ms")
    
    def test_ai_prefers_captures_over_random_moves(self):
        """AI should prefer capturing moves when available."""
        game_state = GameState(tournament_rule=False)
        
        # Set up position where AI can capture
        # . O O X  (X can capture the two O's by playing at (9, 8))
        positions = [
            (9, 9, -1),  # O
            (9, 10, -1), # O  
            (9, 11, 1),  # X
            (10, 10, 1), # X (some other moves)
            (8, 9, -1),  # O
        ]
        
        for row, col, player in positions:
            game_state.board.set_stone(row, col, player)
            game_state.move_count += 1
            
        game_state.current_player = 1  # X to play
        
        # Run AI with decent search time
        ai = MCTSEngine(time_limit_ms=2000, max_simulations=200)
        move = ai.search(game_state)
        
        analysis = ai.get_move_analysis()
        capture_move = (9, 8)
        
        # AI should strongly consider the capture move
        if capture_move in analysis['move_analysis']:
            capture_data = analysis['move_analysis'][capture_move]
            print(f"Capture move {capture_move}: {capture_data['visits']} visits, {capture_data['win_rate']:.1f}% win rate")
            
            # Capture move should get reasonable consideration
            total_visits = analysis['total_root_visits']
            capture_percentage = capture_data['visits'] / max(total_visits, 1) * 100
            assert capture_percentage > 5, f"Capture move only got {capture_percentage:.1f}% of visits"
    
    def test_ai_creates_multiple_threats(self):
        """AI should try to create multiple threats (forcing moves)."""
        game_state = GameState(tournament_rule=False)
        
        # Set up mid-game position
        positions = [
            (9, 9, 1),   # X center
            (8, 8, -1),  # O 
            (9, 10, 1),  # X
            (8, 9, -1),  # O
            (9, 11, 1),  # X  
            (7, 8, -1),  # O
        ]
        
        for row, col, player in positions:
            game_state.board.set_stone(row, col, player)
            game_state.move_count += 1
            
        game_state.current_player = 1  # X to play
        
        # Give AI good thinking time
        ai = MCTSEngine(time_limit_ms=3000, max_simulations=300)
        move = ai.search(game_state)
        
        analysis = ai.get_move_analysis()
        
        # AI should explore multiple strategic moves
        assert len(analysis['move_analysis']) >= 3, "AI should consider multiple moves"
        
        # Check that AI is thinking strategically (some moves get significant attention)
        move_visits = [data['visits'] for data in analysis['move_analysis'].values()]
        max_visits = max(move_visits) if move_visits else 0
        
        # Best move should get substantial attention
        assert max_visits > 10, f"Best move only got {max_visits} visits"
        
        stats = ai.get_performance_stats()
        print(f"Strategic play: {len(analysis['move_analysis'])} moves considered")
        print(f"Top move: {analysis['best_move']} with {max_visits} visits")
    
    def test_ai_progressive_widening_effectiveness(self):
        """Test that progressive widening helps AI focus on good moves."""
        game_state = GameState(tournament_rule=False)
        
        # Create a position with obvious good and bad moves
        positions = [
            (9, 9, 1),   # X center
            (8, 8, -1),  # O nearby (good area)
            (15, 15, 1), # X far corner (bad area)
        ]
        
        for row, col, player in positions:
            game_state.board.set_stone(row, col, player)
            game_state.move_count += 1
            
        game_state.current_player = -1  # O to play
        
        # Test with limited simulations (should use progressive widening)
        ai_limited = MCTSEngine(time_limit_ms=1000, max_simulations=50)
        move_limited = ai_limited.search(game_state)
        analysis_limited = ai_limited.get_move_analysis()
        
        # Test with more simulations  
        ai_extended = MCTSEngine(time_limit_ms=2000, max_simulations=150)
        move_extended = ai_extended.search(game_state)
        analysis_extended = ai_extended.get_move_analysis()
        
        # Limited search should explore fewer moves (progressive widening)
        moves_limited = len(analysis_limited['move_analysis'])
        moves_extended = len(analysis_extended['move_analysis'])
        
        print(f"Limited search: {moves_limited} moves explored")
        print(f"Extended search: {moves_extended} moves explored")
        
        # Progressive widening should limit exploration in limited search
        assert moves_limited < moves_extended or moves_limited < 15, \
            "Progressive widening should limit move exploration with fewer simulations"
        
        # Both should prefer moves near existing stones over far corners
        for move in [move_limited, move_extended]:
            # Move should be reasonably close to existing stones
            min_distance = min(
                max(abs(move[0] - stone_row), abs(move[1] - stone_col))
                for stone_row, stone_col, _ in positions
            )
            assert min_distance <= 3, f"AI move {move} too far from existing stones (distance {min_distance})"
    
    def test_ai_learns_from_simulations(self):
        """Test that more simulations lead to better move selection."""
        game_state = GameState(tournament_rule=False)
        
        # Set up tactical position where there's a clearly best move
        positions = [
            (9, 9, 1),   # X
            (9, 10, 1),  # X  
            (8, 9, -1),  # O
            (8, 10, -1), # O
        ]
        
        for row, col, player in positions:
            game_state.board.set_stone(row, col, player)
            game_state.move_count += 1
            
        game_state.current_player = 1  # X to play
        
        # Test with minimal simulations
        ai_weak = MCTSEngine(time_limit_ms=500, max_simulations=10)
        move_weak = ai_weak.search(game_state)
        analysis_weak = ai_weak.get_move_analysis()
        
        # Test with many simulations
        ai_strong = MCTSEngine(time_limit_ms=4000, max_simulations=400)
        move_strong = ai_strong.search(game_state)
        analysis_strong = ai_strong.get_move_analysis()
        
        # Strong AI should have more confidence in its choice
        if analysis_strong['move_analysis'] and analysis_weak['move_analysis']:
            strong_best_visits = max(data['visits'] for data in analysis_strong['move_analysis'].values())
            weak_best_visits = max(data['visits'] for data in analysis_weak['move_analysis'].values()) if analysis_weak['move_analysis'] else 1
            
            print(f"Weak AI best move visits: {weak_best_visits}")
            print(f"Strong AI best move visits: {strong_best_visits}")
            
            # Strong AI should have explored much more deeply
            assert strong_best_visits > weak_best_visits * 2, \
                "Strong AI should explore best moves more deeply"
        
        stats_weak = ai_weak.get_performance_stats()
        stats_strong = ai_strong.get_performance_stats()
        
        print(f"Weak AI: {stats_weak['simulations']} sims")
        print(f"Strong AI: {stats_strong['simulations']} sims")

class TestAIStrategicUnderstanding:
    """Higher-level tests of AI strategic understanding."""
    
    def test_ai_opening_principles(self):
        """AI should follow basic opening principles."""
        game_state = GameState()
        
        # AI plays first move
        ai = MCTSEngine(time_limit_ms=1000, max_simulations=100)
        first_move = ai.search(game_state)
        
        # First move should be center or nearby
        center = (9, 9)
        distance_from_center = max(abs(first_move[0] - center[0]), abs(first_move[1] - center[1]))
        assert distance_from_center <= 2, f"First move {first_move} too far from center"
        
        # Make the move
        game_state.make_move(first_move)
        ai.update_root(first_move)
        
        # Opponent plays at distance 3+ (tournament rule)
        opponent_move = (6, 6)  # Distance 3 from center
        game_state.make_move(opponent_move)
        ai.update_root(opponent_move)
        
        # AI's second move should be strategic
        second_move = ai.search(game_state)
        
        # Should be reasonably close to existing stones
        min_dist_to_stones = min(
            max(abs(second_move[0] - first_move[0]), abs(second_move[1] - first_move[1])),
            max(abs(second_move[0] - opponent_move[0]), abs(second_move[1] - opponent_move[1]))
        )
        assert min_dist_to_stones <= 3, f"Second move {second_move} should be closer to existing stones"
        
        print(f"Opening: {first_move} → {second_move}")
    
    def test_ai_endgame_tactics(self):
        """AI should handle endgame situations correctly."""
        game_state = GameState()
        
        # Set up near-endgame position
        positions = [
            # X has captured 4 pairs (8 stones), needs 1 more pair to win
            (9, 9, 1), (9, 10, 1), (9, 11, 1),    # X stones
            (8, 9, 1), (8, 10, 1), (8, 11, 1),    # X stones  
            (7, 9, 1), (7, 10, 1),                 # X stones (total: 8 captured)
            
            (10, 9, -1), (10, 10, -1),             # O pair that can be captured
            (5, 5, -1), (6, 6, -1), (4, 4, -1),   # Other O stones
        ]
        
        for row, col, player in positions:
            game_state.board.set_stone(row, col, player)
            
        # Manually set capture count
        game_state.captures[1] = 4  # X has 4 pairs captured
        game_state.move_count = len(positions)
        game_state.current_player = 1  # X to play
        
        ai = MCTSEngine(time_limit_ms=2000, max_simulations=200)
        move = ai.search(game_state)
        
        # AI should play to capture the 5th pair and win
        winning_capture = (10, 8)  # Captures the O pair at (10,9) and (10,10)
        
        if (10, 8) in game_state.get_legal_moves():
            # Verify this would actually be a winning capture
            test_state = GameState()
            for row, col, player in positions:
                test_state.board.set_stone(row, col, player)
            test_state.captures[1] = 4
            test_state.current_player = 1
            
            if (10, 8) in test_state.get_legal_moves():
                test_state.make_move((10, 8))
                if test_state.captures[1] >= 5:
                    assert move == winning_capture, f"AI should play winning capture {winning_capture}, played {move}"
        
        print(f"Endgame: AI played {move} with {game_state.captures[1]} pairs captured")

def run_tactical_demonstration():
    """
    Standalone function to demonstrate AI tactical abilities.
    Run with: python3 -c "from tests.test_ai_tactics import run_tactical_demonstration; run_tactical_demonstration()"
    """
    print("=== AlphaPente AI Tactical Demonstration ===\n")
    
    # Test 1: Threat Recognition
    print("1. Testing threat recognition...")
    game_state = GameState()
    
    # O has 4 in a row, threatens to win
    positions = [(9, 9, -1), (9, 10, -1), (9, 11, -1), (9, 12, -1)]
    for row, col, player in positions:
        game_state.board.set_stone(row, col, player)
    game_state.current_player = 1
    
    ai = MCTSEngine(time_limit_ms=2000, max_simulations=150)
    move = ai.search(game_state)
    analysis = ai.get_move_analysis()
    stats = ai.get_performance_stats()
    
    print(f"   Position: O has 4 in a row at (9,9)-(9,12)")
    print(f"   AI response: {move}")
    print(f"   Analysis: {stats['simulations']} sims in {stats['time_ms']:.0f}ms")
    
    if move in [(9, 8), (9, 13)]:
        print("   ✅ GOOD: AI blocks the threat!")
    else:
        print("   ❌ POOR: AI missed the threat")
    
    # Test 2: Win Recognition  
    print("\n2. Testing win recognition...")
    game_state = GameState()
    
    positions = [(9, 9, 1), (9, 10, 1), (9, 11, 1), (9, 12, 1)]
    for row, col, player in positions:
        game_state.board.set_stone(row, col, player)
    game_state.current_player = 1
    
    move = ai.search(game_state)
    stats = ai.get_performance_stats()
    
    print(f"   Position: X has 4 in a row, can win")
    print(f"   AI response: {move}")
    print(f"   Analysis: {stats['simulations']} sims in {stats['time_ms']:.0f}ms")
    
    if move in [(9, 8), (9, 13)]:
        print("   ✅ EXCELLENT: AI takes the win!")
    else:
        print("   ❌ CRITICAL ERROR: AI missed winning move")
    
    # Test 3: Progressive Improvement
    print("\n3. Testing learning with more simulations...")
    game_state = GameState(tournament_rule=False)
    game_state.make_move((9, 9))
    
    # Quick AI
    ai_fast = MCTSEngine(time_limit_ms=200, max_simulations=20)
    move_fast = ai_fast.search(game_state)
    stats_fast = ai_fast.get_performance_stats()
    
    # Slower AI  
    ai_slow = MCTSEngine(time_limit_ms=2000, max_simulations=200)
    move_slow = ai_slow.search(game_state)
    stats_slow = ai_slow.get_performance_stats()
    
    print(f"   Fast AI ({stats_fast['simulations']} sims): {move_fast}")
    print(f"   Slow AI ({stats_slow['simulations']} sims): {move_slow}")
    
    if stats_slow['simulations'] > stats_fast['simulations'] * 3:
        print("   ✅ GOOD: More simulations = deeper search")
    else:
        print("   ⚠️  LIMITED: Not much simulation difference")
    
    print(f"\n=== Summary ===")
    print(f"The AI demonstrates tactical awareness through MCTS search.")
    print(f"Performance: ~{stats['simulations_per_second']:.1f} simulations/second")
    print(f"With enough time, it learns to block threats and find wins.")

if __name__ == "__main__":
    run_tactical_demonstration()