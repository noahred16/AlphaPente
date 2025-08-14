import pytest
import torch
import numpy as np
from models.gomoku_simple_nn import GomokuSimpleNN
from mcts import MCTS, MCTSNode


class TestMCTSNNIntegration:
    """Test integration between MCTS and neural network model."""
    
    @pytest.fixture
    def model(self):
        """Create a test model."""
        return GomokuSimpleNN(board_size=7)
    
    @pytest.fixture
    def mcts(self, model):
        """Create MCTS instance with model."""
        return MCTS(model, simulations=10)
    
    @pytest.fixture
    def empty_board(self):
        """Create empty 7x7 board."""
        return np.zeros((7, 7), dtype=np.float32)
    
    def test_model_creation(self, model):
        """Test model can be created successfully."""
        assert isinstance(model, GomokuSimpleNN)
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 0
    
    def test_model_forward_pass(self, model):
        """Test model forward pass with correct shapes."""
        batch_size = 2
        board = torch.randn(batch_size, 2, 7, 7)
        player_caps = torch.tensor([2, 1])
        opp_caps = torch.tensor([1, 3])
        
        policy, value = model(board, player_caps, opp_caps)
        
        assert policy.shape == (batch_size, 49)  # 7x7 flattened
        assert value.shape == (batch_size, 1)
    
    def test_model_predict_policy_and_value(self, model, empty_board):
        """Test the predict_policy_and_value method used by MCTS."""
        policy, value = model.predict_policy_and_value(empty_board, 0, 0)
        
        assert policy.shape == (7, 7)
        assert isinstance(value, float)
        assert -1 <= value <= 1
        assert np.isclose(policy.sum(), 1.0, atol=1e-6)  # Should sum to 1
    
    def test_mcts_model_integration(self, mcts, empty_board):
        """Test MCTS can use the model for policy and value prediction."""
        # Test that MCTS can get policy and value from model
        policy, value = mcts.get_policy_value(empty_board, 0, 0)
        
        assert policy.shape == (7, 7)
        assert isinstance(value, float)
        assert -1 <= value <= 1
    
    def test_mcts_node_creation_with_model(self, mcts, empty_board):
        """Test MCTSNode can be created with model predictions."""
        # Get initial prediction
        policy, value = mcts.get_policy_value(empty_board, 0, 0)
        
        # Create root node
        root = MCTSNode(
            board=empty_board,
            policy_prior=None,
            value=value,
            player_captures=0,
            opponent_captures=0,
            num_moves=0
        )
        
        assert root.value == value
        assert not root.is_terminal
        assert len(root.untried_moves) > 0
    
    def test_mcts_expansion_uses_model(self, mcts, empty_board):
        """Test MCTS expansion uses model for child node values."""
        root = MCTSNode(
            board=empty_board,
            policy_prior=None,
            value=0.0,
            player_captures=0,
            opponent_captures=0,
            num_moves=0
        )
        
        # Test expansion
        child = mcts.expansion(root)
        
        assert child is not None
        assert len(root.children) > 0
        assert child.policy_prior is not None
        assert isinstance(child.value, float)
    
    def test_mcts_full_simulation(self, mcts, empty_board):
        """Test full MCTS simulation with model."""
        # Run a small number of simulations
        result = mcts.run(empty_board, 0, 0)
        
        assert result is not None
        assert isinstance(result, MCTSNode)
        assert result.num_visits > 0
    
    def test_mcts_best_move(self, mcts, empty_board):
        """Test MCTS can select best move using model."""
        move, policy = mcts.best_move(empty_board, 0, 0)
        
        assert move is not None
        assert len(move) == 2  # (row, col)
        assert 0 <= move[0] < 7
        assert 0 <= move[1] < 7
        assert policy.shape == (7, 7)
        assert np.isclose(policy.sum(), 1.0, atol=1e-6)