"""ResNet-Style Neural Network Architecture for AlphaPente/Gomoku

This module implements an improved neural network architecture using residual connections
for better gradient flow and representation learning in the AlphaPente game.

Key Architecture Features:
- Residual blocks with skip connections for deeper training
- Separate optimized policy and value heads
- Global average pooling to reduce overfitting
- Batch normalization for training stability
- Gradient clipping and weight decay for robust training

Architecture Overview:
    Input (2×7×7) → Initial Conv(128) → ResBlock(128) → ResBlock(128)
    ├── Policy Head: Conv(32) → Conv(1) → Flatten → Softmax 
    └── Value Head: Conv(32) → Global Pool → FC + Captures → Tanh

Improvements over Simple CNN:
- ~2x more parameters for increased capacity
- Skip connections enable deeper networks without vanishing gradients
- Separate head architectures optimized for policy vs value prediction
- Better handling of spatial features through residual learning

Compatibility:
- Same interface as GomokuSimpleNN for drop-in replacement
- Works with existing MCTS implementation
- Supports batch prediction and training methods

Usage:
    model = GomokuResNet(board_size=7, num_residual_blocks=2)
    policy, value = model.predict_policy_and_value(board, p_caps, o_caps)
    
Training:
    batch = {...}  # Training data
    losses = model.train_on_batch(batch, epochs=1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualBlock(nn.Module):
    """
    Basic residual block with skip connection.
    
    Architecture: Input → Conv → BN → ReLU → Conv → BN → (+Input) → ReLU
    
    The skip connection allows gradients to flow directly through the network,
    enabling deeper architectures and better learning of complex patterns.
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class GomokuResNet(nn.Module):
    def __init__(self, board_size=7, num_residual_blocks=2, channels=128):
        super(GomokuResNet, self).__init__()
        self.board_size = board_size
        self.channels = channels
        
        # Initial convolutional layer
        input_channels = 2  # player channel + opponent channel
        self.initial_conv = nn.Conv2d(input_channels, channels, kernel_size=3, padding=1, bias=False)
        self.initial_bn = nn.BatchNorm2d(channels)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(channels) for _ in range(num_residual_blocks)
        ])
        
        # Policy head
        self.policy_conv1 = nn.Conv2d(channels, 32, kernel_size=1, bias=False)
        self.policy_bn1 = nn.BatchNorm2d(32)
        self.policy_conv2 = nn.Conv2d(32, 1, kernel_size=1)
        self.policy_flatten = nn.Flatten()
        
        # Value head
        self.value_conv = nn.Conv2d(channels, 32, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_pool = nn.AdaptiveAvgPool2d(1)
        self.value_flatten = nn.Flatten()
        
        # Capture features integration
        capture_input_size = 2  # player_captures, opponent_captures
        self.value_fc1 = nn.Linear(32 + capture_input_size, 64)
        self.value_fc2 = nn.Linear(64, 1)
        
    def forward(self, board_state, player_captures, opponent_captures):
        """
        Forward pass through the network.
        Args:
            board_state: tensor of shape (batch_size, 2, 7, 7)
            player_captures: tensor of shape (batch_size,)
            opponent_captures: tensor of shape (batch_size,)
        Returns:
            policy: tensor of shape (batch_size, 49) - raw logits for each board position
            value: tensor of shape (batch_size, 1) - position evaluation between -1 and 1
        """
        # Initial convolution
        x = F.relu(self.initial_bn(self.initial_conv(board_state)))
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Policy head
        policy = F.relu(self.policy_bn1(self.policy_conv1(x)))
        policy = self.policy_conv2(policy)
        policy = self.policy_flatten(policy)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = self.value_pool(value)
        value = self.value_flatten(value)
        
        # Integrate capture features
        player_captures = player_captures.view(-1, 1).float() / 10.0
        opponent_captures = opponent_captures.view(-1, 1).float() / 10.0
        value = torch.cat([value, player_captures, opponent_captures], dim=1)
        
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value
    
    def predict(self, board_state, player_captures, opponent_captures, valid_moves_mask=None):
        """
        Make predictions for a single board state.
        Args:
            board_state: tensor of shape (2, 7, 7)
            player_captures: scalar or tensor with player's capture count
            opponent_captures: scalar or tensor with opponent's capture count
            valid_moves_mask: optional tensor of shape (7, 7) with 1 for valid moves
        Returns:
            move_probs: tensor of shape (49,) with probabilities for each move
            value: scalar tensor with position evaluation
        """
        # Add batch dimension
        board_state = board_state.unsqueeze(0)
        player_captures = (
            torch.tensor([player_captures])
            if not torch.is_tensor(player_captures)
            else player_captures.unsqueeze(0)
        )
        opponent_captures = (
            torch.tensor([opponent_captures])
            if not torch.is_tensor(opponent_captures)
            else opponent_captures.unsqueeze(0)
        )
        
        # Get predictions
        policy_logits, value = self.forward(
            board_state, player_captures, opponent_captures
        )
        
        # Reshape policy to board shape
        policy_logits = policy_logits.view(-1)
        
        # Apply valid moves mask if provided
        if valid_moves_mask is not None:
            valid_moves_flat = valid_moves_mask.view(-1)
            policy_logits = policy_logits.masked_fill(valid_moves_flat == 0, -1e9)
        
        # Convert to probabilities
        move_probs = F.softmax(policy_logits, dim=0)
        return move_probs, value.squeeze()
    
    def train_on_batch(self, batch, epochs=1):
        """
        Train the network on a batch of game positions.
        Args:
            batch: dict with keys:
                - 'states': tensor of shape (batch_size, 2, 7, 7)
                - 'player_captures': tensor of shape (batch_size,)
                - 'opponent_captures': tensor of shape (batch_size,)
                - 'policies': tensor of shape (batch_size, 49)
                - 'values': tensor of shape (batch_size,)
            epochs: number of times to train on this batch
        Returns:
            dict with average losses over all epochs
        """
        self.train()
        
        # Extract batch data
        states = batch["states"]
        player_captures = batch["player_captures"]
        opponent_captures = batch["opponent_captures"]
        target_policies = batch["policies"]
        target_values = batch["values"]
        
        # Ensure target_values has correct shape
        if target_values.dim() == 1:
            target_values = target_values.unsqueeze(1)
        
        # Initialize optimizer if not exists
        if not hasattr(self, "optimizer"):
            self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-4)
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        
        for epoch in range(epochs):
            # Forward pass
            policy_logits, predicted_values = self.forward(
                states, player_captures, opponent_captures
            )
            
            # Calculate policy loss (KL divergence)
            target_policies_sum = target_policies.sum(dim=1, keepdim=True)
            target_policies_sum = torch.clamp(target_policies_sum, min=1e-8)
            target_policies_norm = target_policies / target_policies_sum
            target_policies_norm = target_policies_norm.view(policy_logits.shape)
            
            policy_loss = F.kl_div(
                F.log_softmax(policy_logits, dim=1),
                target_policies_norm,
                reduction="batchmean",
            )
            
            # Calculate value loss (MSE)
            value_loss = F.mse_loss(predicted_values, target_values)
            
            # Combined loss
            total_loss = policy_loss + value_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Track losses
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
        
        # Return average losses
        return {
            "policy_loss": total_policy_loss / epochs,
            "value_loss": total_value_loss / epochs,
            "total_loss": (total_policy_loss + total_value_loss) / epochs,
        }
    
    def predict_policy_and_value(self, board, player_captures, opponent_captures, debug=False):
        """
        Predict policy and value for a given game state.
        Args:
            board: np.array of shape (7, 7) with 1 for player, -1 for opponent, 0 for empty
            player_captures: int - number of captures by player
            opponent_captures: int - number of captures by opponent
        Returns:
            policy: np.array of shape (7, 7) with move probabilities
            value: float - position evaluation between -1 and 1
        """
        if debug:
            value = np.random.uniform(-1, 1)
            policy = np.ones((7, 7)) / 49.0
            return policy, value
        
        # Set model to evaluation mode
        self.eval()
        torch.manual_seed(42)
        
        # Preprocess the game state
        board_tensor, player_cap_tensor, opponent_cap_tensor = preprocess_game_state(
            board, player_captures, opponent_captures
        )
        
        # Get legal moves mask
        valid_moves_mask = torch.tensor(board == 0, dtype=torch.float32)
        
        # Make prediction without computing gradients
        with torch.no_grad():
            move_probs, value = self.predict(
                board_tensor, player_cap_tensor, opponent_cap_tensor, valid_moves_mask
            )
        
        # Reshape policy back to board shape
        policy = move_probs.view(7, 7).numpy()
        value = value.item()
        
        return policy, value
    
    def predict_batch(self, game_states):
        """
        Predict policies and values for a batch of game states.
        Args:
            game_states: list of tuples (board, player_captures, opponent_captures)
        Returns:
            policies: list of np.arrays of shape (7, 7) with move probabilities
            values: list of floats - position evaluations between -1 and 1
        """
        self.eval()
        
        if not game_states:
            return [], []
        
        # Preprocess all game states
        board_tensors = []
        player_cap_tensors = []
        opponent_cap_tensors = []
        valid_moves_masks = []
        
        for board, player_captures, opponent_captures in game_states:
            board_tensor, player_cap_tensor, opponent_cap_tensor = (
                preprocess_game_state(board, player_captures, opponent_captures)
            )
            board_tensors.append(board_tensor)
            player_cap_tensors.append(player_cap_tensor)
            opponent_cap_tensors.append(opponent_cap_tensor)
            
            valid_moves_mask = torch.tensor(board == 0, dtype=torch.float32)
            valid_moves_masks.append(valid_moves_mask)
        
        # Stack into batch tensors
        batch_boards = torch.stack(board_tensors)
        batch_player_caps = torch.stack(player_cap_tensors)
        batch_opponent_caps = torch.stack(opponent_cap_tensors)
        batch_valid_moves = torch.stack(valid_moves_masks)
        
        # Make predictions without computing gradients
        with torch.no_grad():
            policy_logits, values = self.forward(
                batch_boards, batch_player_caps, batch_opponent_caps
            )
            
            # Apply valid moves masking and softmax for each item in batch
            batch_size = len(game_states)
            policies = []
            
            for i in range(batch_size):
                logits = policy_logits[i]
                valid_mask_flat = batch_valid_moves[i].view(-1)
                masked_logits = logits.masked_fill(valid_mask_flat == 0, -1e9)
                move_probs = F.softmax(masked_logits, dim=0)
                policy = move_probs.view(7, 7).numpy()
                policies.append(policy)
            
            values_list = values.squeeze().tolist()
            if not isinstance(values_list, list):
                values_list = [values_list]
        
        return policies, values_list


def preprocess_game_state(board, player_captures, opponent_captures):
    """
    Convert game state to tensors suitable for the neural network.
    Args:
        board: np.array of shape (7, 7) with 1 for player, -1 for opponent, 0 for empty
        player_captures: int - number of captures by player
        opponent_captures: int - number of captures by opponent
    Returns:
        board_tensor: torch.tensor of shape (2, 7, 7)
        player_captures_tensor: torch.tensor scalar
        opponent_captures_tensor: torch.tensor scalar
    """
    # Create two-channel board representation
    player_channel = (board == 1).astype(np.float32)
    opponent_channel = (board == -1).astype(np.float32)
    
    # Stack channels
    board_channels = np.stack([player_channel, opponent_channel], axis=0)
    
    # Convert to tensors
    board_tensor = torch.tensor(board_channels, dtype=torch.float32)
    player_captures_tensor = torch.tensor(player_captures, dtype=torch.float32)
    opponent_captures_tensor = torch.tensor(opponent_captures, dtype=torch.float32)
    
    return board_tensor, player_captures_tensor, opponent_captures_tensor