import torch
import torch.nn as nn
import torch.nn.functional as F


class GomokuSimpleNN(nn.Module):  # Class name remains consistent
    def __init__(self, board_size):
        super(GomokuSimpleNN, self).__init__()
        self.board_size = board_size
        input_channels = 3  # board, player_captures, opponent_captures

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        # Fully connected layers for policy head
        self.policy_conv = nn.Conv2d(128, 2, kernel_size=1)
        self.policy_fc = nn.Linear(
            2 * board_size[0] * board_size[1], board_size[0] * board_size[1]
        )

        # Fully connected layers for value head
        self.value_conv = nn.Conv2d(128, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(board_size[0] * board_size[1], 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, board, player_captures, opponent_captures, num_moves):
        # Combine inputs into a single tensor
        captures = torch.full_like(
            board, player_captures, dtype=torch.float32
        ).unsqueeze(1)
        opponent = torch.full_like(
            board, opponent_captures, dtype=torch.float32
        ).unsqueeze(1)
        x = torch.cat([board.unsqueeze(1), captures, opponent], dim=1)

        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Policy head
        policy = F.relu(self.policy_conv(x))
        policy = policy.view(policy.size(0), -1)
        policy = F.softmax(self.policy_fc(policy), dim=1)

        # Value head
        value = F.relu(self.value_conv(x))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value
