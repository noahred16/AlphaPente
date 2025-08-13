import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GomokuSimpleNN(nn.Module):
    def __init__(self, board_size=7, hidden_size=128):
        super(GomokuSimpleNN, self).__init__()
        self.board_size = board_size
        # Input: board state (1 channel for current player, 1 for opponent)
        # Can be extended to more channels for additional features
        input_channels = 2
        # Convolutional layers to extract spatial features
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # Batch normalization for stability
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        # Calculate size for fully connected layer
        conv_output_size = 128 * board_size * board_size
        # Additional input for capture counts (2 values: player_captures, opponent_captures)
        capture_input_size = 2
        # Combined input size for shared layer
        combined_input_size = conv_output_size + capture_input_size
        # Shared fully connected layer
        self.fc_shared = nn.Linear(combined_input_size, hidden_size)
        # Policy head (predicts move probabilities)
        self.policy_head = nn.Linear(hidden_size, board_size * board_size)
        # Value head (predicts position evaluation)
        self.value_hidden = nn.Linear(hidden_size, 64)
        self.value_head = nn.Linear(64, 1)

    def forward(self, board_state, player_captures, opponent_captures):
        """
        Forward pass through the network.
        Args:
            board_state: tensor of shape (batch_size, 2, 7, 7)
                - Channel 0: current player stones (1 where player has stone, 0 elsewhere)
                - Channel 1: opponent stones (1 where opponent has stone, 0 elsewhere)
            player_captures: tensor of shape (batch_size,) - number of captures by player
            opponent_captures: tensor of shape (batch_size,) - number of captures by opponent
        Returns:
            policy: tensor of shape (batch_size, 49) - raw logits for each board position
            value: tensor of shape (batch_size, 1) - position evaluation between -1 and 1
        """
        # Convolutional layers with ReLU and batch norm
        x = F.relu(self.bn1(self.conv1(board_state)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        # Ensure captures have correct shape and normalize
        player_captures = (
            player_captures.view(-1, 1).float() / 10.0
        )  # Assuming max ~10 captures
        opponent_captures = opponent_captures.view(-1, 1).float() / 10.0
        # Concatenate conv features with capture counts
        combined = torch.cat([x, player_captures, opponent_captures], dim=1)
        # Shared representation
        shared = F.relu(self.fc_shared(combined))
        # Policy output (raw logits for each board position)
        policy = self.policy_head(shared)
        # Value output (position evaluation between -1 and 1)
        value = F.relu(self.value_hidden(shared))
        value = torch.tanh(self.value_head(value))
        return policy, value

    def predict(
        self, board_state, player_captures, opponent_captures, valid_moves_mask=None
    ):
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
            # Set invalid moves to very negative value
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
                - 'player_captures': tensor of shape (batch_size,) - player capture counts
                - 'opponent_captures': tensor of shape (batch_size,) - opponent capture counts
                - 'policies': tensor of shape (batch_size, 49) - target move distributions
                - 'values': tensor of shape (batch_size,) - target position evaluations
            epochs: number of times to train on this batch
        Returns:
            dict with average losses over all epochs
        """
        self.train()  # Set to training mode
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
            self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        total_policy_loss = 0.0
        total_value_loss = 0.0
        for epoch in range(epochs):
            # Forward pass
            policy_logits, predicted_values = self.forward(
                states, player_captures, opponent_captures
            )
            # Calculate policy loss (KL divergence)
            # Ensure target_policies is normalized and avoid division by zero
            target_policies_sum = target_policies.sum(dim=1, keepdim=True)
            target_policies_sum = torch.clamp(
                target_policies_sum, min=1e-8
            )  # Avoid zero division
            target_policies_norm = target_policies / target_policies_sum
            target_policies_norm = target_policies_norm.view(
                policy_logits.shape
            )  # Match shape
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

    def predict_policy_and_value(
        self, board, player_captures, opponent_captures, debug=False
    ):
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
        # just random move for debug
        if debug:
            value = np.random.uniform(-1, 1)
            policy = np.ones((7, 7)) / 49.0
            return policy, value
        # Set model to evaluation mode
        self.eval()
        torch.manual_seed(42)  # Set a fixed seed
        # Preprocess the game state
        board_tensor, player_cap_tensor, opponent_cap_tensor = preprocess_game_state(
            board, player_captures, opponent_captures
        )
        # Get legal moves mask (assuming empty spaces are legal moves)
        valid_moves_mask = torch.tensor(board == 0, dtype=torch.float32)
        # Make prediction without computing gradients
        with torch.no_grad():
            move_probs, value = self.predict(
                board_tensor, player_cap_tensor, opponent_cap_tensor, valid_moves_mask
            )
        # Reshape policy back to board shape
        policy = move_probs.view(7, 7).numpy()
        # Convert value to float
        value = value.item()
        return policy, value

    def predict_batch(self, game_states):
        """
        Predict policies and values for a batch of game states.

        Args:
            game_states: list of tuples (board, player_captures, opponent_captures)
                - board: np.array of shape (7, 7) with 1 for player, -1 for opponent, 0 for empty
                - player_captures: int - number of captures by player
                - opponent_captures: int - number of captures by opponent

        Returns:
            policies: list of np.arrays of shape (7, 7) with move probabilities
            values: list of floats - position evaluations between -1 and 1
        """
        # Set model to evaluation mode
        self.eval()

        # Handle empty batch
        if not game_states:
            return [], []

        # Preprocess all game states
        board_tensors = []
        player_cap_tensors = []
        opponent_cap_tensors = []
        valid_moves_masks = []

        for board, player_captures, opponent_captures in game_states:
            # Preprocess each game state
            board_tensor, player_cap_tensor, opponent_cap_tensor = (
                preprocess_game_state(board, player_captures, opponent_captures)
            )
            board_tensors.append(board_tensor)
            player_cap_tensors.append(player_cap_tensor)
            opponent_cap_tensors.append(opponent_cap_tensor)

            # Create valid moves mask
            valid_moves_mask = torch.tensor(board == 0, dtype=torch.float32)
            valid_moves_masks.append(valid_moves_mask)

        # Stack into batch tensors
        batch_boards = torch.stack(board_tensors)
        batch_player_caps = torch.stack(player_cap_tensors)
        batch_opponent_caps = torch.stack(opponent_cap_tensors)
        batch_valid_moves = torch.stack(valid_moves_masks)

        # Make predictions without computing gradients
        with torch.no_grad():
            # Get raw predictions from forward pass
            policy_logits, values = self.forward(
                batch_boards, batch_player_caps, batch_opponent_caps
            )

            # Apply valid moves masking and softmax for each item in batch
            batch_size = len(game_states)
            policies = []

            for i in range(batch_size):
                # Get logits for this position
                logits = policy_logits[i]

                # Apply valid moves mask
                valid_mask_flat = batch_valid_moves[i].view(-1)
                masked_logits = logits.masked_fill(valid_mask_flat == 0, -1e9)

                # Convert to probabilities
                move_probs = F.softmax(masked_logits, dim=0)

                # Reshape to board shape and convert to numpy
                policy = move_probs.view(7, 7).numpy()
                policies.append(policy)

            # Convert values to list of floats
            values_list = values.squeeze().tolist()

            # Handle case where batch_size is 1 (values might be a scalar)
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
