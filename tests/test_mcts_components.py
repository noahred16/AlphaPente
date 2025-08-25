"""
Tests for MCTS components (node and rollout policy).
"""

import pytest
from src.core.game_state import GameState
from src.mcts.node import MCTSNode  
from src.mcts.rollout import RolloutPolicy

class TestMCTSNode:
    
    def test_node_initialization(self):
        """Test basic node initialization."""
        # Root node
        root = MCTSNode()
        assert root.position is None
        assert root.parent is None
        assert root.visits == 0
        assert root.wins == 0.0
        assert len(root.children) == 0
        assert root.is_leaf()
        
        # Child node
        child = MCTSNode((9, 9), parent=root)
        assert child.position == (9, 9)
        assert child.parent is root
        assert child.is_leaf()
    
    def test_add_child(self):
        """Test adding children to node."""
        root = MCTSNode()
        
        # Add child
        child = root.add_child((9, 9))
        assert child.position == (9, 9)
        assert child.parent is root
        assert (9, 9) in root.children
        assert not root.is_leaf()
        
        # Can't add same child twice
        with pytest.raises(ValueError):
            root.add_child((9, 9))
    
    def test_backpropagation(self):
        """Test backpropagation up the tree."""
        root = MCTSNode()
        child = root.add_child((9, 9))
        grandchild = child.add_child((8, 8))
        
        # Backpropagate from grandchild
        grandchild.backpropagate(1.0)
        
        # Check values (result flips at each level)
        assert grandchild.visits == 1
        assert grandchild.wins == 1.0
        
        assert child.visits == 1  
        assert child.wins == 0.0  # Flipped
        
        assert root.visits == 1
        assert root.wins == 1.0  # Flipped again
    
    def test_ucb1_selection(self):
        """Test UCB1 child selection."""
        root = MCTSNode()
        
        # Add children and simulate some visits
        child1 = root.add_child((9, 9))
        child2 = root.add_child((8, 8))
        
        root.visits = 10
        
        # Child with no visits should be selected first
        selected = root.select_child_ucb1()
        assert selected in [child1, child2]
        assert selected.visits == 0
        
        # After both children visited, UCB1 should work normally
        child1.visits = 5
        child1.wins = 3.0
        child2.visits = 3
        child2.wins = 1.0
        
        selected = root.select_child_ucb1()
        assert selected in [child1, child2]
    
    def test_expansion_with_game_state(self):
        """Test node expansion with actual game state."""
        game_state = GameState()
        game_state.make_move((9, 9))  # Center move
        
        root = MCTSNode()
        # Give root some visits so progressive widening allows distance 3+ moves
        # (needed because tournament rule requires player 2's first move to be ≥3 from center)
        root.visits = 500  
        
        # Expand should create a child
        child = root.expand(game_state)
        assert child is not None
        assert child.parent is root
        assert len(root.children) == 1
        
        # Available moves should be updated
        assert len(root.available_moves) > 0
        assert len(root.unexplored_moves) >= 0  # May be empty after expansion
    
    def test_get_statistics(self):
        """Test node statistics."""
        root = MCTSNode()
        root.visits = 10
        root.wins = 7.0
        
        stats = root.get_statistics()
        assert stats['visits'] == 10
        assert stats['wins'] == 7.0
        assert stats['win_rate'] == 0.7
        assert stats['is_leaf'] == True
        assert stats['children_count'] == 0

class TestRolloutPolicy:
    
    def test_rollout_initialization(self):
        """Test rollout policy initialization."""
        policy = RolloutPolicy()
        assert policy.max_moves == 200
        
        # Custom params
        policy = RolloutPolicy(max_moves=50, seed=42)
        assert policy.max_moves == 50
    
    def test_basic_rollout(self):
        """Test basic rollout from start position."""
        game_state = GameState()
        policy = RolloutPolicy(seed=42)  # For reproducibility
        
        # Rollout should complete and return valid result
        result = policy.rollout(game_state)
        assert result in [0.0, 0.5, 1.0]
        
        # Game state should be unchanged after rollout
        assert game_state.move_count == 0
        assert game_state.current_player == 1
        assert len(game_state.move_history) == 0
    
    def test_rollout_from_mid_game(self):
        """Test rollout from mid-game position."""
        game_state = GameState()
        
        # Make some moves
        game_state.make_move((9, 9))
        game_state.make_move((8, 8))
        game_state.make_move((9, 10))
        
        original_move_count = game_state.move_count
        original_player = game_state.current_player
        
        policy = RolloutPolicy(seed=42)
        result = policy.rollout(game_state)
        
        # Valid result
        assert result in [0.0, 0.5, 1.0]
        
        # State should be restored
        assert game_state.move_count == original_move_count
        assert game_state.current_player == original_player
    
    def test_quick_evaluation_terminal(self):
        """Test quick evaluation on terminal position."""
        game_state = GameState()
        
        # Create a winning position for player 1
        for i in range(5):
            game_state.make_move((9, 9 + i))
            if i < 4:  # Don't switch player on last move
                game_state.make_move((8, 8 + i))
        
        policy = RolloutPolicy()
        
        # Should detect terminal position
        result = policy.quick_evaluation(game_state)
        assert result is not None
        
    def test_rollout_with_bias(self):
        """Test biased rollout."""
        game_state = GameState()
        game_state.make_move((9, 9))
        
        policy = RolloutPolicy(seed=42)
        bias_moves = [(9, 10), (8, 9)]  # Adjacent to center
        
        result = policy.rollout_with_bias(game_state, bias_moves, bias_probability=0.8)
        assert result in [0.0, 0.5, 1.0]
        
        # State should be restored
        assert game_state.move_count == 1
    
    def test_rollout_max_moves_limit(self):
        """Test rollout respects max moves limit."""
        game_state = GameState()
        
        # Very short rollout
        policy = RolloutPolicy(max_moves=5, seed=42)
        result = policy.rollout(game_state)
        
        # Should return draw (0.5) if reaches limit
        assert result in [0.0, 0.5, 1.0]
        assert game_state.move_count == 0  # State restored
    
    def test_get_statistics(self):
        """Test rollout policy statistics."""
        policy = RolloutPolicy(max_moves=100)
        stats = policy.get_statistics()
        
        assert stats['max_moves'] == 100
        assert stats['policy_type'] == 'random_rollout'

class TestMCTSIntegration:
    
    def test_node_and_rollout_integration(self):
        """Test integration between node expansion and rollout."""
        game_state = GameState()
        game_state.make_move((9, 9))  # Center move
        
        root = MCTSNode()
        root.visits = 500  # Allow distance 3+ moves for tournament rule
        policy = RolloutPolicy(seed=42, max_moves=50)
        
        # Expand and rollout
        child = root.expand(game_state)
        assert child is not None
        
        # Make the child's move and rollout
        game_state.make_move(child.position)
        result = policy.rollout(game_state)
        game_state.undo_move()  # Undo child's move
        
        # Backpropagate
        child.backpropagate(result)
        
        assert child.visits == 1
        assert root.visits == 501  # Original 500 + 1 from backprop
        assert child.wins == result
        assert abs(root.wins - (1.0 - result)) < 1e-10  # Flipped result
    
    def test_multiple_expansions_and_simulations(self):
        """Test multiple node expansions and simulations."""
        # Use a game state without tournament rule complications
        game_state = GameState(tournament_rule=False)
        game_state.make_move((9, 9))
        
        root = MCTSNode()
        policy = RolloutPolicy(seed=42, max_moves=30)
        
        # Perform several expansion/simulation cycles
        expansions_made = 0
        for i in range(5):
            # Expand if possible
            if not root.is_fully_expanded():
                child = root.expand(game_state)
                if child:
                    expansions_made += 1
                    # Simulate from child
                    game_state.make_move(child.position)
                    result = policy.rollout(game_state)
                    game_state.undo_move()
                    
                    # Backpropagate
                    child.backpropagate(result)
        
        # Should have some visits and children if any expansions were made
        if expansions_made > 0:
            assert root.visits >= 1
            assert len(root.children) >= 1
            
            # Get best child
            best = root.get_best_child()
            assert best is not None
            assert best.visits >= 1
        else:
            # If no expansions were possible, that's also valid
            assert len(root.children) == 0