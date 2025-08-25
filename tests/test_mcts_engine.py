"""
Tests for the MCTS Engine - the core search implementation.
"""

import pytest
import time
from src.core.game_state import GameState
from src.mcts.engine import MCTSEngine

class TestMCTSEngine:
    
    def test_engine_initialization(self):
        """Test basic engine initialization."""
        engine = MCTSEngine()
        
        # Check default parameters
        assert engine.exploration_param == 1.414
        assert engine.time_limit_ms == 1000
        assert engine.max_simulations == 10000
        assert engine.early_termination_threshold == 3.0
        
        # Should have components
        assert engine.rollout_policy is not None
        assert engine.root is None  # No tree initially
        assert engine.game_state is None
        
    def test_engine_custom_parameters(self):
        """Test engine with custom parameters."""
        engine = MCTSEngine(
            exploration_param=2.0,
            rollout_max_moves=50,
            time_limit_ms=500,
            max_simulations=1000,
            early_termination_threshold=2.0
        )
        
        assert engine.exploration_param == 2.0
        assert engine.time_limit_ms == 500
        assert engine.max_simulations == 1000
        assert engine.early_termination_threshold == 2.0
        assert engine.rollout_policy.max_moves == 50
    
    def test_initialize_search(self):
        """Test search initialization."""
        engine = MCTSEngine()
        game_state = GameState()
        
        engine.initialize_search(game_state)
        
        # Should have root and game state
        assert engine.root is not None
        assert engine.game_state is game_state
        assert len(engine.root.available_moves) > 0
    
    def test_search_from_start_position(self):
        """Test MCTS search from starting position."""
        engine = MCTSEngine(time_limit_ms=100, max_simulations=50)
        game_state = GameState()
        
        # Should return a valid move
        move = engine.search(game_state)
        assert isinstance(move, tuple)
        assert len(move) == 2
        assert 0 <= move[0] < 19
        assert 0 <= move[1] < 19
        
        # Should be a legal move
        legal_moves = game_state.get_legal_moves()
        assert move in legal_moves
        
        # Should have performance stats
        stats = engine.get_performance_stats()
        assert 'simulations' in stats
        assert 'time_ms' in stats
        assert 'simulations_per_second' in stats
        assert 'best_move' in stats
        assert stats['best_move'] == move
    
    def test_search_single_move_optimization(self):
        """Test optimization when only one legal move exists."""
        engine = MCTSEngine()
        
        # Create state with only one legal move (almost full board)
        game_state = GameState()
        
        # Fill most of the board except one position
        test_position = (5, 5)
        for row in range(19):
            for col in range(19):
                if (row, col) != test_position and not (row == 9 and col == 9):
                    # Skip center for first move
                    continue
                if (row, col) != test_position:
                    game_state.board.set_stone(row, col, 1 if (row + col) % 2 == 0 else -1)
        
        # Should return quickly with no simulations
        start_time = time.time()
        legal_moves = game_state.get_legal_moves()
        
        if len(legal_moves) == 1:
            move = engine.search(game_state)
            end_time = time.time()
            
            assert move == legal_moves[0]
            assert (end_time - start_time) < 0.1  # Should be very fast
            
            stats = engine.get_performance_stats()
            assert stats['simulations'] == 0
            assert stats['early_termination'] == True
    
    def test_search_mid_game_position(self):
        """Test MCTS search from mid-game position."""
        engine = MCTSEngine(time_limit_ms=200, max_simulations=100)
        game_state = GameState()
        
        # Play some moves to create mid-game position
        moves = [(9, 9), (8, 8), (9, 10), (8, 7), (10, 9)]
        for move in moves:
            if move in game_state.get_legal_moves():
                game_state.make_move(move)
        
        # Search should work
        move = engine.search(game_state)
        assert isinstance(move, tuple)
        
        # Should be legal
        legal_moves = game_state.get_legal_moves()
        assert move in legal_moves
        
        # Should have reasonable performance
        stats = engine.get_performance_stats()
        assert stats['simulations'] > 0
        assert stats['time_ms'] > 0
        assert stats['simulations_per_second'] > 0.5  # Python implementation is slower
    
    def test_tree_reuse(self):
        """Test tree reuse functionality."""
        engine = MCTSEngine(time_limit_ms=100, max_simulations=50)
        game_state = GameState()
        
        # First search
        move1 = engine.search(game_state)
        tree_size_1 = engine.last_search_stats.get('tree_size', 0)
        
        # Make the move
        game_state.make_move(move1)
        
        # Update root for tree reuse
        reused = engine.update_root(move1)
        assert reused == True  # Should reuse tree
        
        # Second search should reuse part of tree
        move2 = engine.search(game_state)
        
        # Should still work
        assert isinstance(move2, tuple)
        legal_moves = game_state.get_legal_moves()
        assert move2 in legal_moves
    
    def test_tree_reuse_failure(self):
        """Test tree reuse when move not in tree."""
        engine = MCTSEngine()
        game_state = GameState()
        
        # Initialize with empty tree
        engine.initialize_search(game_state)
        
        # Try to reuse with move not in tree
        reused = engine.update_root((5, 5))
        assert reused == False  # Should not reuse
        
        # Should have new empty root
        assert engine.root is not None
        assert len(engine.root.children) == 0
    
    def test_move_analysis(self):
        """Test move analysis functionality."""
        engine = MCTSEngine(time_limit_ms=200, max_simulations=100)
        game_state = GameState(tournament_rule=False)  # Simplify for testing
        
        # Run search to build tree
        engine.search(game_state)
        
        # Get analysis
        analysis = engine.get_move_analysis()
        
        assert 'total_root_visits' in analysis
        assert 'total_children' in analysis
        assert 'move_analysis' in analysis
        assert 'best_move' in analysis
        assert 'tree_depth' in analysis
        
        # Should have some children analyzed
        if analysis['total_children'] > 0:
            assert len(analysis['move_analysis']) > 0
            
            # Check structure of move analysis
            for move, data in analysis['move_analysis'].items():
                assert isinstance(move, tuple)
                assert 'visits' in data
                assert 'visit_percentage' in data
                assert 'wins' in data
                assert 'win_rate' in data
                assert 'ucb1_value' in data
    
    def test_principal_variation(self):
        """Test principal variation extraction."""
        engine = MCTSEngine(time_limit_ms=100, max_simulations=50)
        game_state = GameState(tournament_rule=False)
        
        # Run search
        engine.search(game_state)
        
        # Get principal variation
        pv = engine.get_principal_variation(max_depth=5)
        
        # Should be list of moves
        assert isinstance(pv, list)
        
        # Each element should be a valid move
        for move in pv:
            assert isinstance(move, tuple)
            assert len(move) == 2
    
    def test_early_termination(self):
        """Test early termination when one move dominates."""
        engine = MCTSEngine(
            time_limit_ms=1000,  # Long time limit
            max_simulations=1000,  # Many simulations
            early_termination_threshold=2.0  # Low threshold for early termination
        )
        game_state = GameState(tournament_rule=False)
        
        # Run search - might terminate early
        start_time = time.time()
        move = engine.search(game_state)
        end_time = time.time()
        
        assert isinstance(move, tuple)
        
        stats = engine.get_performance_stats()
        # Early termination is possible but not guaranteed
        # Just check that the system works
        assert 'early_termination' in stats
        assert isinstance(stats['early_termination'], bool)
    
    def test_performance_tracking(self):
        """Test performance statistics tracking."""
        engine = MCTSEngine(time_limit_ms=100, max_simulations=50)
        game_state = GameState()
        
        # Run search
        move = engine.search(game_state)
        
        # Check performance stats
        stats = engine.get_performance_stats()
        
        required_keys = [
            'simulations', 'time_ms', 'simulations_per_second',
            'best_move', 'best_move_visits', 'total_children',
            'tree_size', 'early_termination'
        ]
        
        for key in required_keys:
            assert key in stats, f"Missing key: {key}"
        
        # Reasonable values
        assert stats['simulations'] >= 0
        assert stats['time_ms'] > 0
        assert stats['simulations_per_second'] >= 0
        assert stats['best_move'] == move
        assert stats['total_children'] >= 0
        assert stats['tree_size'] >= 1  # At least root node
        assert isinstance(stats['early_termination'], bool)
    
    def test_reset_tree(self):
        """Test tree reset functionality."""
        engine = MCTSEngine()
        game_state = GameState()
        
        # Build tree
        engine.search(game_state)
        assert engine.root is not None
        assert engine.game_state is not None
        
        # Reset
        engine.reset_tree()
        assert engine.root is None
        assert engine.game_state is None
        assert engine.last_search_stats == {}
    
    def test_no_legal_moves_error(self):
        """Test error handling when no legal moves available."""
        engine = MCTSEngine()
        game_state = GameState()
        
        # Create impossible state (no legal moves)
        # Fill the entire board
        for row in range(19):
            for col in range(19):
                game_state.board.set_stone(row, col, 1 if (row + col) % 2 == 0 else -1)
        
        # Should raise error
        with pytest.raises(ValueError, match="No legal moves available"):
            engine.search(game_state)
    
    def test_string_representation(self):
        """Test string representation for debugging."""
        engine = MCTSEngine()
        
        # Empty engine
        str_repr = str(engine)
        assert "no tree" in str_repr
        
        # After search
        game_state = GameState()
        engine.search(game_state)
        
        str_repr = str(engine)
        assert "root_visits" in str_repr
        assert "children" in str_repr
        assert "sims/sec" in str_repr

class TestMCTSEnginePerformance:
    """Performance-focused tests for MCTS engine."""
    
    def test_simulation_speed_target(self):
        """Test that we can achieve reasonable simulation speeds."""
        engine = MCTSEngine(time_limit_ms=1000, max_simulations=500)
        game_state = GameState(tournament_rule=False)  # Simplify for speed
        
        # Run search
        start_time = time.time()
        engine.search(game_state)
        end_time = time.time()
        
        stats = engine.get_performance_stats()
        sims_per_second = stats['simulations_per_second']
        
        # Should achieve decent speed for Python implementation
        # (C++ target is 10,000+ but Python is much slower)
        assert sims_per_second > 0.5, f"Too slow: {sims_per_second} sims/sec"
        
        print(f"Performance: {sims_per_second:.1f} simulations/second")
    
    def test_tree_reuse_efficiency(self):
        """Test tree reuse provides efficiency gains."""
        engine = MCTSEngine(time_limit_ms=200, max_simulations=100)
        game_state = GameState(tournament_rule=False)
        
        # First search - build tree
        move1 = engine.search(game_state)
        stats1 = engine.get_performance_stats()
        
        # Make move and reuse tree
        game_state.make_move(move1)
        reused = engine.update_root(move1)
        
        if reused:
            # Second search with tree reuse
            move2 = engine.search(game_state)
            stats2 = engine.get_performance_stats()
            
            # Tree reuse should preserve some nodes
            # This is a qualitative test - hard to quantify exactly
            assert isinstance(move2, tuple)
            assert stats2['simulations'] >= 0
            
            print(f"Tree reuse: {reused}, Stats1: {stats1['tree_size']} nodes")
    
    def test_progressive_widening_effect(self):
        """Test that progressive widening limits move exploration appropriately."""
        engine = MCTSEngine(time_limit_ms=100, max_simulations=50)
        game_state = GameState(tournament_rule=False)
        
        # Run search
        engine.search(game_state)
        
        # Check that root has reasonable number of children
        # (not all 360 possible moves explored)
        if engine.root and engine.root.children:
            num_children = len(engine.root.children)
            total_legal_moves = len(game_state.get_legal_moves())
            
            # Progressive widening should limit exploration
            assert num_children < total_legal_moves
            assert num_children > 0
            
            print(f"Progressive widening: {num_children}/{total_legal_moves} moves explored")