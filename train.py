from game import (
    reset_game,
    get_legal_moves,
    make_move,
    pretty_print,
    check_n_in_a_row,
    get_all_orientations,
)
from settings_loader import (
    BOARD_SIZE,
    TABLE_NAME,
    DB_PATH,
    TOURNAMENT_RULES_ENABLED,
    MODEL,
)
from db import Database
from models.gomoku_simple_nn import GomokuSimpleNN, preprocess_game_state
import random
import torch
import os
import numpy as np
from mcts import MCTSNode, MCTS

# Initialize the database
db = Database(DB_PATH, BOARD_SIZE)

# Model and training settings
model = GomokuSimpleNN(BOARD_SIZE[0])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
checkpoint_path = f"./checkpoints/{MODEL}.pt"

# Create checkpoint directory if it doesn't exist
os.makedirs("./checkpoints", exist_ok=True)

# Load checkpoint if it exists
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    print(f"Loaded checkpoint from {checkpoint_path}")

# Training parameters
batch_size = 32  # Increased for better gradient estimates
num_epochs = 100
save_interval = 10  # Save checkpoint every N epochs

# Training loop - always starts from epoch 0
for epoch in range(num_epochs):
    # Fetch training data
    training_data = db.fetch_collection(TABLE_NAME, 1000)  # Get most recent 10k records

    if not training_data:
        print(f"No training data available")
        break

    # Shuffle data for this epoch
    random.shuffle(training_data)

    # Track epoch losses
    epoch_policy_loss = 0.0
    epoch_value_loss = 0.0
    num_batches = 0

    # Process in batches
    for batch_start in range(0, len(training_data), batch_size):
        batch_end = min(batch_start + batch_size, len(training_data))
        batch_data = training_data[batch_start:batch_end]

        # Prepare batch tensors
        board_states = []
        player_captures_list = []
        opponent_captures_list = []
        policies = []
        values = []

        for data in batch_data:
            id, board, player_captures, opponent_captures, num_moves, policy, value = (
                db.decode_row(data)
            )

            # turn and flip board and policy. 16 possible orientations
            boards_orientations = get_all_orientations(board)
            policies_orientations = get_all_orientations(policy)

            for i, (board_oriented, policy_oriented) in enumerate(
                zip(boards_orientations, policies_orientations)
            ):
                # Preprocess the game state
                board_tensor, player_cap_tensor, opponent_cap_tensor = (
                    preprocess_game_state(
                        board_oriented, player_captures, opponent_captures
                    )
                )
                board_states.append(board_tensor)
                player_captures_list.append(player_cap_tensor)
                opponent_captures_list.append(opponent_cap_tensor)

                # Convert policy and value to tensors
                # Policy should be a flat tensor of shape (49,) for 7x7 board
                # Make a copy to avoid negative stride issues
                policy_tensor = torch.tensor(
                    policy_oriented.copy(), dtype=torch.float32
                )
                value_tensor = torch.tensor(value, dtype=torch.float32)

                policies.append(policy_tensor)
                values.append(value_tensor)

        # Stack batch tensors
        batch = {
            "states": torch.stack(board_states),
            "player_captures": torch.stack(player_captures_list),
            "opponent_captures": torch.stack(opponent_captures_list),
            "policies": torch.stack(policies),
            "values": torch.stack(values),
        }

        # Validate batch tensors
        for key, tensor in batch.items():
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Invalid values detected in batch tensor: {key}")

        # Train on batch
        losses = model.train_on_batch(batch, epochs=1)

        # Accumulate losses
        epoch_policy_loss += losses["policy_loss"]
        epoch_value_loss += losses["value_loss"]
        num_batches += 1

        # Print progress every 100 batches
        if (batch_start // batch_size + 1) % 100 == 0:
            print(
                f"Epoch {epoch}, Batch {batch_start // batch_size + 1}/{len(training_data) // batch_size}: "
                f"Policy Loss: {losses['policy_loss']:.4f}, Value Loss: {losses['value_loss']:.4f}"
            )

    # Calculate average losses for epoch
    avg_policy_loss = epoch_policy_loss / num_batches if num_batches > 0 else 0
    avg_value_loss = epoch_value_loss / num_batches if num_batches > 0 else 0

    print(f"\nEpoch {epoch} completed:")
    print(f"  Average Policy Loss: {avg_policy_loss:.4f}")
    print(f"  Average Value Loss: {avg_value_loss:.4f}")
    print(f"  Total samples trained: {len(training_data)}")

    # Save checkpoint periodically (without epoch tracking)
    if (epoch + 1) % save_interval == 0 or epoch == num_epochs - 1:
        checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "policy_loss": avg_policy_loss,
            "value_loss": avg_value_loss,
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch + 1}")

print("\nTraining completed!")
