"""
Tests for MoveGenerator with distance ordering and progressive widening.
"""

import pytest
from src.core.move_generator import MoveGenerator, get_move_generator
from src.core.game_state import GameState

class TestMoveGenerator:
    """Test suite for MoveGenerator functionality."""
    
    def test_initialization(self):
        """Test MoveGenerator initialization."""
        generator = MoveGenerator(board_size=19)
        
        assert generator.board_size == 19
        assert generator.distance_rings is not None
        assert generator.distance_rings.board_size == 19
    
    def test_get_ordered_moves_empty_board(self):
        """Test move generation on empty board."""
        generator = MoveGenerator(board_size=5)
        game = GameState(board_size=5, tournament_rule=False)
        
        # Empty board should return all positions
        moves = generator.get_ordered_moves(game, max_distance=3)
        assert len(moves) == 25  # 5x5 board
        
        # All positions should be legal
        legal_moves = set(game.get_legal_moves())
        for move in moves:
            assert move in legal_moves
    
    def test_get_ordered_moves_single_stone(self):
        """Test move generation with single stone on board."""
        generator = MoveGenerator(board_size=7)
        game = GameState(board_size=7, tournament_rule=False)
        
        # Place stone at center
        game.make_move((3, 3))
        
        # Get moves at distance 1
        moves = generator.get_ordered_moves(game, max_distance=1)
        
        # Should return positions adjacent to center, excluding center itself
        expected_adjacent = [
            (2, 2), (2, 3), (2, 4),
            (3, 2),         (3, 4),
            (4, 2), (4, 3), (4, 4)
        ]
        
        assert len(moves) == 8
        for move in moves:
            assert move in expected_adjacent
        
        # Center should not be included (occupied)
        assert (3, 3) not in moves
    
    def test_get_ordered_moves_multiple_stones(self):
        """Test move generation with multiple stones."""
        generator = MoveGenerator(board_size=7)
        game = GameState(board_size=7, tournament_rule=False)
        
        # Place stones
        game.make_move((2, 2))
        game.make_move((4, 4))
        
        moves = generator.get_ordered_moves(game, max_distance=2)
        
        # Should not include occupied positions
        assert (2, 2) not in moves
        assert (4, 4) not in moves
        
        # Should include positions near both stones
        assert len(moves) > 0
        
        # All moves should be legal
        legal_moves = set(game.get_legal_moves())
        for move in moves:
            assert move in legal_moves
    
    def test_get_ordered_moves_distance_ordering(self):
        """Test that moves are properly ordered by distance."""
        generator = MoveGenerator(board_size=7)
        game = GameState(board_size=7, tournament_rule=False)
        
        # Place stone at center
        game.make_move((3, 3))
        
        # Get moves up to distance 2
        moves = generator.get_ordered_moves(game, max_distance=2)
        
        # First 8 moves should be at distance 1 from center
        distance_1_positions = [
            (2, 2), (2, 3), (2, 4),
            (3, 2),         (3, 4),
            (4, 2), (4, 3), (4, 4)
        ]
        
        for i in range(min(8, len(moves))):
            move = moves[i]
            # Calculate distance from center
            distance = max(abs(move[0] - 3), abs(move[1] - 3))
            assert distance == 1, f"Move {move} should be at distance 1, got {distance}"
    
    def test_get_ordered_moves_respects_tournament_rule(self):
        """Test that tournament rule is respected."""
        generator = MoveGenerator(board_size=19)
        game = GameState(board_size=19, tournament_rule=True)
        
        # Make first move at center
        game.make_move((9, 9))
        
        # Second player's move should respect tournament rule
        moves = generator.get_ordered_moves(game, max_distance=5)
        
        # All moves should be ≥3 from center
        center = 9
        for move in moves:
            row, col = move
            distance = max(abs(row - center), abs(col - center))
            assert distance >= 3, f"Move {move} violates tournament rule (distance {distance} < 3)"
    
    def test_progressive_widening_thresholds(self):
        """Test progressive widening visit thresholds."""
        generator = MoveGenerator(board_size=9)
        game = GameState(board_size=9, tournament_rule=False)
        
        # Add some stones for interesting move generation
        game.make_move((4, 4))
        game.make_move((4, 5))
        game.make_move((5, 4))
        
        # Test different visit thresholds
        early_moves = generator.get_progressive_widening_moves(game, node_visits=5)
        medium_moves = generator.get_progressive_widening_moves(game, node_visits=50)
        deep_moves = generator.get_progressive_widening_moves(game, node_visits=500)
        full_moves = generator.get_progressive_widening_moves(game, node_visits=5000)
        
        # Should have increasing number of moves with more visits
        # (unless we hit the board boundary)
        assert len(early_moves) <= len(medium_moves)
        assert len(medium_moves) <= len(deep_moves)
        
        # Early exploration should be limited
        assert len(early_moves) <= 20  # Target is ~15
        
        # All moves should be legal
        legal_moves = set(game.get_legal_moves())
        for moves_list in [early_moves, medium_moves, deep_moves, full_moves]:
            for move in moves_list:
                assert move in legal_moves
    
    def test_progressive_widening_targets(self):
        """Test that progressive widening hits approximately the target move counts."""
        generator = MoveGenerator(board_size=19)
        game = GameState(board_size=19, tournament_rule=False)
        
        # Create interesting position with several stones
        moves_to_make = [(9, 9), (10, 10), (8, 8), (11, 11), (7, 7)]
        for move in moves_to_make:
            game.make_move(move)
        
        # Test progressive widening targets
        early = generator.get_progressive_widening_moves(game, 5)    # Target ~15
        medium = generator.get_progressive_widening_moves(game, 50)  # Target ~30
        deep = generator.get_progressive_widening_moves(game, 500)   # Target ~50
        full = generator.get_progressive_widening_moves(game, 5000) # Target ~80
        
        print(f"Progressive widening: early={len(early)}, medium={len(medium)}, deep={len(deep)}, full={len(full)}")
        
        # Should be reasonable approximations of targets
        # (allowing some variance since board constraints matter)
        assert 8 <= len(early) <= 25    # Target 15
        assert 15 <= len(medium) <= 45   # Target 30
        assert 25 <= len(deep) <= 75     # Target 50
    
    def test_get_immediate_moves(self):
        """Test getting immediate tactical moves."""
        generator = MoveGenerator(board_size=7)
        game = GameState(board_size=7, tournament_rule=False)
        
        # Place stones
        game.make_move((3, 3))
        game.make_move((5, 5))
        
        immediate = generator.get_immediate_moves(game)
        
        # Should only include distance-1 moves
        for move in immediate:
            row, col = move
            # Check distance to both stones
            dist_to_first = max(abs(row - 3), abs(col - 3))
            dist_to_second = max(abs(row - 5), abs(col - 5))
            min_distance = min(dist_to_first, dist_to_second)
            assert min_distance == 1, f"Move {move} not at distance 1 (min distance: {min_distance})"
    
    def test_get_all_legal_moves_ordered(self):
        """Test getting all legal moves in distance order."""
        generator = MoveGenerator(board_size=5)
        game = GameState(board_size=5, tournament_rule=False)
        
        # Place a stone
        game.make_move((2, 2))
        
        all_moves = generator.get_all_legal_moves_ordered(game)
        legal_moves = game.get_legal_moves()
        
        # Should have same number of moves
        assert len(all_moves) == len(legal_moves)
        
        # Should include all legal moves
        for move in legal_moves:
            assert move in all_moves
        
        # Should be ordered by distance (closer first)
        prev_min_distance = 0
        for move in all_moves:
            row, col = move
            distance_to_stone = max(abs(row - 2), abs(col - 2))
            assert distance_to_stone >= prev_min_distance
            # Allow ties (same distance), but don't allow going backwards
            if distance_to_stone > prev_min_distance:
                prev_min_distance = distance_to_stone
    
    def test_estimate_move_count_at_distance(self):
        """Test estimation of moves at specific distances."""
        generator = MoveGenerator(board_size=7)
        game = GameState(board_size=7, tournament_rule=False)
        
        # Empty board
        count = generator.estimate_move_count_at_distance(game, 1)
        assert count == 0  # No stones, so no moves at any distance
        
        # With one stone at center
        game.make_move((3, 3))
        count_d1 = generator.estimate_move_count_at_distance(game, 1)
        count_d2 = generator.estimate_move_count_at_distance(game, 2)
        
        assert count_d1 == 8  # 8 neighbors around center
        assert count_d2 > 0   # Some positions at distance 2
    
    def test_get_move_statistics(self):
        """Test move generation statistics."""
        generator = MoveGenerator(board_size=7)
        game = GameState(board_size=7, tournament_rule=False)
        
        # Add some stones
        game.make_move((3, 3))
        game.make_move((4, 4))
        
        stats = generator.get_move_statistics(game)
        
        # Check required fields
        assert 'total_legal_moves' in stats
        assert 'stone_count' in stats
        assert 'moves_by_distance' in stats
        assert 'progressive_widening' in stats
        
        assert stats['stone_count'] == 2
        assert stats['total_legal_moves'] == 47  # 49 - 2 occupied
        
        # Distance statistics
        assert isinstance(stats['moves_by_distance'], dict)
        assert 1 in stats['moves_by_distance']
        
        # Progressive widening statistics
        pw_stats = stats['progressive_widening']
        assert 'early_exploration' in pw_stats
        assert 'medium_exploration' in pw_stats
        assert 'deep_exploration' in pw_stats
        assert 'full_exploration' in pw_stats
        
        # Should show increasing move counts
        assert pw_stats['early_exploration'] <= pw_stats['medium_exploration']
    
    def test_global_instance(self):
        """Test global move generator instance."""
        # First call creates instance
        gen1 = get_move_generator(7)
        assert isinstance(gen1, MoveGenerator)
        assert gen1.board_size == 7
        
        # Second call returns same instance
        gen2 = get_move_generator(7)
        assert gen2 is gen1
        
        # Different size creates new instance
        gen3 = get_move_generator(9)
        assert gen3 is not gen1
        assert gen3.board_size == 9
    
    def test_performance_ordering(self):
        """Test that move ordering performs reasonably well."""
        generator = MoveGenerator(board_size=19)
        game = GameState(board_size=19, tournament_rule=False)
        
        # Create a complex mid-game position
        moves_to_make = [
            (9, 9), (10, 10), (8, 8), (11, 11), (7, 7),
            (12, 12), (6, 6), (13, 13), (5, 5), (14, 14)
        ]
        for move in moves_to_make:
            game.make_move(move)
        
        import time
        
        # Time multiple move generation calls
        start_time = time.time()
        for _ in range(100):
            moves = generator.get_ordered_moves(game, max_distance=3)
        end_time = time.time()
        
        avg_time_us = (end_time - start_time) * 1000000 / 100
        print(f"Average move generation time: {avg_time_us:.1f} μs")
        
        # Should be fast (target < 1 μs from spec, but allow more for Python)
        assert avg_time_us < 1000  # 1000 μs = 1ms (generous for Python)
        
        # Should generate reasonable number of moves
        assert len(moves) > 0
        assert len(moves) < 200  # Shouldn't be all moves
    
    def test_empty_history_handling(self):
        """Test handling of games with no move history."""
        generator = MoveGenerator(board_size=5)
        game = GameState(board_size=5, tournament_rule=False)
        
        # No moves made yet
        moves = generator.get_ordered_moves(game, max_distance=3)
        assert len(moves) == 25  # All positions legal
        
        # Progressive widening should still work
        pw_moves = generator.get_progressive_widening_moves(game, 50)
        assert len(pw_moves) == 25  # All positions
    
    def test_tournament_rule_integration(self):
        """Test integration with tournament rule."""
        generator = MoveGenerator(board_size=19)
        game = GameState(board_size=19, tournament_rule=True)
        
        # First move - should get all positions
        moves = generator.get_ordered_moves(game, max_distance=5)
        assert len(moves) == 361
        
        # Make center move
        game.make_move((9, 9))
        
        # Second player should be restricted
        moves = generator.get_ordered_moves(game, max_distance=5)
        
        # No moves should be close to center
        for move in moves:
            row, col = move
            distance = max(abs(row - 9), abs(col - 9))
            assert distance >= 3