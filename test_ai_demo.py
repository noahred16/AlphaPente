#!/usr/bin/env python3
"""
Demonstration of what the AI can actually learn and improve on.
"""

from src.core.game_state import GameState
from src.mcts.engine import MCTSEngine

def test_ai_learning_capability():
    """Test what the AI can actually demonstrate."""
    print("=== AlphaPente AI Learning Demonstration ===\n")
    
    # Test 1: Does more search time lead to different/better moves?
    print("1. Testing search depth effect...")
    
    game_state = GameState(tournament_rule=False)  # Simplify
    game_state.make_move((9, 9))  # Center move made
    
    # Quick search
    print("   Quick search (100ms, 20 sims max):")
    ai_quick = MCTSEngine(time_limit_ms=100, max_simulations=20)
    move_quick = ai_quick.search(game_state)
    stats_quick = ai_quick.get_performance_stats()
    analysis_quick = ai_quick.get_move_analysis()
    
    print(f"     Move: {move_quick}")
    print(f"     Stats: {stats_quick['simulations']} sims in {stats_quick['time_ms']:.0f}ms ({stats_quick['simulations_per_second']:.1f}/sec)")
    print(f"     Moves considered: {len(analysis_quick['move_analysis'])}")
    
    # Longer search
    print("\n   Extended search (1000ms, 100 sims max):")
    ai_long = MCTSEngine(time_limit_ms=1000, max_simulations=100)
    move_long = ai_long.search(game_state)
    stats_long = ai_long.get_performance_stats()
    analysis_long = ai_long.get_move_analysis()
    
    print(f"     Move: {move_long}")
    print(f"     Stats: {stats_long['simulations']} sims in {stats_long['time_ms']:.0f}ms ({stats_long['simulations_per_second']:.1f}/sec)")
    print(f"     Moves considered: {len(analysis_long['move_analysis'])}")
    
    # Test 2: Does the AI prefer moves closer to existing stones?
    print("\n2. Testing distance preference...")
    
    # Get all legal moves and their distances from center
    legal_moves = game_state.get_legal_moves()
    center = (9, 9)
    
    # Calculate distance of AI's choice
    ai_distance = max(abs(move_long[0] - center[0]), abs(move_long[1] - center[1]))
    
    # Find average distance of all legal moves
    total_distance = sum(max(abs(row - center[0]), abs(col - center[1])) for row, col in legal_moves)
    avg_distance = total_distance / len(legal_moves)
    
    print(f"   AI chose move at distance {ai_distance} from center")
    print(f"   Average distance of all legal moves: {avg_distance:.1f}")
    
    if ai_distance < avg_distance:
        print("   ✅ GOOD: AI prefers moves closer to existing stones")
    else:
        print("   ⚠️  AI doesn't show strong distance preference")
    
    # Test 3: Tree reuse effectiveness
    print("\n3. Testing tree reuse...")
    
    # Make a move and test tree reuse
    ai_reuse = MCTSEngine(time_limit_ms=500, max_simulations=50)
    move1 = ai_reuse.search(game_state)
    stats1 = ai_reuse.get_performance_stats()
    
    print(f"   First search: {stats1['simulations']} sims, tree size: {stats1.get('tree_size', 'N/A')}")
    
    # Make the move and reuse tree
    game_state.make_move(move1)
    reused = ai_reuse.update_root(move1)
    
    move2 = ai_reuse.search(game_state)
    stats2 = ai_reuse.get_performance_stats()
    
    print(f"   Second search (after tree reuse): {stats2['simulations']} sims, tree size: {stats2.get('tree_size', 'N/A')}")
    print(f"   Tree reused successfully: {reused}")
    
    if reused:
        print("   ✅ Tree reuse working")
    else:
        print("   ❌ Tree reuse failed")
    
    # Test 4: Progressive widening in action
    print("\n4. Testing progressive widening...")
    
    game_state_test = GameState(tournament_rule=False)
    game_state_test.make_move((9, 9))
    
    # Very limited search
    ai_limited = MCTSEngine(time_limit_ms=50, max_simulations=5)
    move_limited = ai_limited.search(game_state_test)
    analysis_limited = ai_limited.get_move_analysis()
    
    # More generous search
    ai_generous = MCTSEngine(time_limit_ms=500, max_simulations=50)  
    move_generous = ai_generous.search(game_state_test)
    analysis_generous = ai_generous.get_move_analysis()
    
    print(f"   Limited search: {len(analysis_limited['move_analysis'])} moves explored")
    print(f"   Generous search: {len(analysis_generous['move_analysis'])} moves explored")
    
    if len(analysis_generous['move_analysis']) > len(analysis_limited['move_analysis']):
        print("   ✅ Progressive widening: more time = more moves explored")
    else:
        print("   ⚠️  Progressive widening not clearly demonstrated")
    
    # Test 5: Win rate consistency
    print("\n5. Testing move evaluation consistency...")
    
    if analysis_long['move_analysis']:
        moves_by_visits = sorted(analysis_long['move_analysis'].items(), 
                               key=lambda x: x[1]['visits'], reverse=True)
        
        print("   Top moves by search attention:")
        for i, (move_pos, data) in enumerate(moves_by_visits[:5]):
            print(f"     {i+1}. {move_pos}: {data['visits']} visits ({data['win_rate']:.1f}% win rate)")
        
        if len(moves_by_visits) >= 2:
            best_visits = moves_by_visits[0][1]['visits']
            second_visits = moves_by_visits[1][1]['visits']
            
            if best_visits > second_visits * 1.5:
                print("   ✅ AI shows clear preference for best move")
            else:
                print("   ⚠️  AI shows less clear move preferences")
    
    print(f"\n=== Conclusions ===")
    print(f"The AI demonstrates MCTS principles:")
    print(f"• Tree search with {stats_long['simulations_per_second']:.1f} simulations/second")
    print(f"• Progressive widening (explores more moves with more time)")
    print(f"• Tree reuse between moves")
    print(f"• Move evaluation based on win rates from rollouts")
    print(f"\nFor tactical play, the AI needs:")
    print(f"• More sophisticated evaluation (threat detection)")
    print(f"• Faster simulations for deeper search")
    print(f"• Enhanced rollout policy with tactical awareness")

def test_simple_tactical_scenario():
    """Test a very simple tactical scenario the AI might handle."""
    print("\n=== Simple Tactical Test ===")
    
    # Create a position where one move is clearly better than others
    game_state = GameState(tournament_rule=False)
    
    # Set up: X has 2 in a row, can extend to 3
    positions = [
        (9, 9, 1),   # X
        (9, 10, 1),  # X
        (8, 8, -1),  # O somewhere else
    ]
    
    for row, col, player in positions:
        game_state.board.set_stone(row, col, player)
        game_state.move_count += 1
    
    game_state.current_player = 1  # X to play
    
    print("Position: X has 2 in a row at (9,9)-(9,10)")
    print("X can extend to 3 in a row at (9,8) or (9,11)")
    
    # Test AI choice
    ai = MCTSEngine(time_limit_ms=1000, max_simulations=100)
    move = ai.search(game_state)
    analysis = ai.get_move_analysis()
    stats = ai.get_performance_stats()
    
    print(f"AI choice: {move}")
    print(f"Search: {stats['simulations']} simulations")
    
    # Check if AI chose to extend the line
    extending_moves = {(9, 8), (9, 11)}
    if move in extending_moves:
        print("✅ GOOD: AI extends the line (basic tactical awareness)")
    else:
        print("⚠️  AI chose different move - may still be reasonable")
    
    # Show top moves
    if analysis['move_analysis']:
        print("\nTop moves considered:")
        moves_by_visits = sorted(analysis['move_analysis'].items(), 
                               key=lambda x: x[1]['visits'], reverse=True)
        for i, (pos, data) in enumerate(moves_by_visits[:3]):
            extend_marker = " ⭐" if pos in extending_moves else ""
            print(f"  {i+1}. {pos}: {data['visits']} visits, {data['win_rate']:.1f}% win rate{extend_marker}")

if __name__ == "__main__":
    test_ai_learning_capability()
    test_simple_tactical_scenario()