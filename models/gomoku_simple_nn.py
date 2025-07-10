import torch
import torch.nn as nn
import torch.nn.functional as F


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

        # Shared fully connected layer
        self.fc_shared = nn.Linear(conv_output_size, hidden_size)

        # Policy head (predicts move probabilities)
        self.policy_head = nn.Linear(hidden_size, board_size * board_size)

        # Value head (predicts position evaluation)
        self.value_hidden = nn.Linear(hidden_size, 64)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        # x shape: (batch_size, 2, 7, 7)
        # Channel 0: current player stones (1 where player has stone, 0 elsewhere)
        # Channel 1: opponent stones (1 where opponent has stone, 0 elsewhere)

        # Convolutional layers with ReLU and batch norm
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # Shared representation
        shared = F.relu(self.fc_shared(x))

        # Policy output (raw logits for each board position)
        policy = self.policy_head(shared)

        # Value output (position evaluation between -1 and 1)
        value = F.relu(self.value_hidden(shared))
        value = torch.tanh(self.value_head(value))

        return policy, value

    def predict(self, board_state, valid_moves_mask=None):
        """
        Make predictions for a single board state.

        Args:
            board_state: tensor of shape (2, 7, 7)
            valid_moves_mask: optional tensor of shape (7, 7) with 1 for valid moves

        Returns:
            move_probs: tensor of shape (49,) with probabilities for each move
            value: scalar tensor with position evaluation
        """
        # Add batch dimension
        x = board_state.unsqueeze(0)

        # Get predictions
        policy_logits, value = self.forward(x)

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
                - 'policies': tensor of shape (batch_size, 49) - target move distributions
                - 'values': tensor of shape (batch_size,) - target position evaluations
            epochs: number of times to train on this batch

        Returns:
            dict with average losses over all epochs
        """
        self.train()  # Set to training mode

        # Extract batch data
        states = batch["states"]
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
            policy_logits, predicted_values = self.forward(states)

            # Calculate policy loss (cross-entropy)
            # Ensure target_policies is normalized
            target_policies_norm = target_policies / target_policies.sum(
                dim=1, keepdim=True
            )
            policy_loss = F.cross_entropy(policy_logits, target_policies_norm)

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


def create_board_state(board, current_player):
    """
    Convert board array to neural network input format.

    Args:
        board: numpy array or tensor of shape (7, 7) with 0=empty, 1=player1, 2=player2
        current_player: 1 or 2

    Returns:
        tensor of shape (2, 7, 7)
    """
    if not isinstance(board, torch.Tensor):
        board = torch.tensor(board, dtype=torch.float32)

    opponent = 3 - current_player  # If player is 1, opponent is 2, and vice versa

    # Create two-channel representation
    current_player_stones = (board == current_player).float()
    opponent_stones = (board == opponent).float()

    return torch.stack([current_player_stones, opponent_stones])
