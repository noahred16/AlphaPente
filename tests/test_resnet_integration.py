import pytest
import torch
import numpy as np
from models.gomoku_resnet import GomokuResNet
from mcts import MCTS, MCTSNode


class TestResNetIntegration:
    """Test integration between MCTS and ResNet model."""
    
    @pytest.fixture
    def resnet_model(self):
        """Create a test ResNet model."""
        return GomokuResNet(board_size=7, num_residual_blocks=2)
    
    @pytest.fixture
    def mcts_resnet(self, resnet_model):
        """Create MCTS instance with ResNet model."""
        return MCTS(resnet_model, simulations=10)
    
    @pytest.fixture
    def empty_board(self):
        """Create empty 7x7 board."""
        return np.zeros((7, 7), dtype=np.float32)
    
    def test_resnet_model_creation(self, resnet_model):
        """Test ResNet model can be created successfully."""
        assert isinstance(resnet_model, GomokuResNet)
        param_count = sum(p.numel() for p in resnet_model.parameters())
        assert param_count > 0
        # Should have more parameters than simple model due to residual blocks
        assert param_count > 100000
    
    def test_resnet_forward_pass(self, resnet_model):
        """Test ResNet model forward pass with correct shapes."""
        batch_size = 2
        board = torch.randn(batch_size, 2, 7, 7)
        player_caps = torch.tensor([2, 1])
        opp_caps = torch.tensor([1, 3])
        
        policy, value = resnet_model(board, player_caps, opp_caps)
        
        assert policy.shape == (batch_size, 49)  # 7x7 flattened
        assert value.shape == (batch_size, 1)
        assert torch.all(value >= -1) and torch.all(value <= 1)
    
    def test_resnet_predict_policy_and_value(self, resnet_model, empty_board):
        """Test the predict_policy_and_value method used by MCTS."""
        policy, value = resnet_model.predict_policy_and_value(empty_board, 0, 0)
        
        assert policy.shape == (7, 7)
        assert isinstance(value, float)
        assert -1 <= value <= 1
        assert np.isclose(policy.sum(), 1.0, atol=1e-6)
    
    def test_resnet_mcts_integration(self, mcts_resnet, empty_board):
        """Test MCTS can use the ResNet model for policy and value prediction."""
        policy, value = mcts_resnet.get_policy_value(empty_board, 0, 0)
        
        assert policy.shape == (7, 7)
        assert isinstance(value, float)
        assert -1 <= value <= 1
    
    def test_resnet_mcts_best_move(self, mcts_resnet, empty_board):
        """Test MCTS can select best move using ResNet model."""
        move, policy = mcts_resnet.best_move(empty_board, 0, 0)
        
        assert move is not None
        assert len(move) == 2
        assert 0 <= move[0] < 7
        assert 0 <= move[1] < 7
        assert policy.shape == (7, 7)
        assert np.isclose(policy.sum(), 1.0, atol=1e-6)
    
    def test_resnet_residual_blocks(self, resnet_model):
        """Test that residual blocks are working correctly."""
        # Test that the model has residual blocks
        assert len(resnet_model.residual_blocks) == 2
        
        # Test forward pass through residual blocks
        x = torch.randn(1, 2, 7, 7)
        caps = torch.tensor([0]), torch.tensor([0])
        
        with torch.no_grad():
            policy, value = resnet_model(x, caps[0], caps[1])
        
        # Should produce valid outputs
        assert not torch.isnan(policy).any()
        assert not torch.isnan(value).any()
    
    def test_resnet_batch_prediction(self, resnet_model):
        """Test batch prediction functionality."""
        # Create batch of game states
        game_states = [
            (np.zeros((7, 7)), 0, 0),
            (np.random.choice([0, 1, -1], size=(7, 7)), 1, 2),
        ]
        
        policies, values = resnet_model.predict_batch(game_states)
        
        assert len(policies) == 2
        assert len(values) == 2
        
        for policy in policies:
            assert policy.shape == (7, 7)
            assert np.isclose(policy.sum(), 1.0, atol=1e-6)
        
        for value in values:
            assert isinstance(value, float)
            assert -1 <= value <= 1
    
    def test_resnet_model_comparison(self, empty_board):
        """Compare ResNet and simple model outputs."""
        from models.gomoku_simple_nn import GomokuSimpleNN
        
        simple_model = GomokuSimpleNN(board_size=7)
        resnet_model = GomokuResNet(board_size=7, num_residual_blocks=2)
        
        # Both should produce valid outputs
        simple_policy, simple_value = simple_model.predict_policy_and_value(empty_board, 0, 0)
        resnet_policy, resnet_value = resnet_model.predict_policy_and_value(empty_board, 0, 0)
        
        # Both should have correct shapes
        assert simple_policy.shape == resnet_policy.shape == (7, 7)
        assert isinstance(simple_value, float) and isinstance(resnet_value, float)
        
        # Both should be valid probability distributions
        assert np.isclose(simple_policy.sum(), 1.0, atol=1e-6)
        assert np.isclose(resnet_policy.sum(), 1.0, atol=1e-6)