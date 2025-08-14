"""Enhanced Model Training Script for AlphaPente Training Pipeline v2

This script trains neural network models using data generated from self-play games.
It features intelligent data sampling strategies to improve training quality and prevent overfitting.

Key Improvements over v1:
- Intelligent sampling strategies (recent, random, top_random)
- Support for multiple model architectures (simple CNN, ResNet)
- Configurable training parameters (batch size, learning rate, epochs)
- Data augmentation with orientation transforms
- JSON configuration with command-line overrides
- Checkpoint management with epoch tracking
- Comprehensive loss monitoring and logging

Sampling Strategies:
1. 'recent': Uses most recent records (fast, may overfit)
2. 'random': Completely random sampling (diverse, may include low-quality data)
3. 'top_random': 70% recent + 30% random older (RECOMMENDED - balances quality/diversity)

How it works:
1. Loads configuration and command-line parameters
2. Fetches training data using specified sampling strategy
3. Applies data augmentation (rotations/reflections)
4. Trains model in batches with policy and value loss
5. Saves checkpoints periodically
6. Provides detailed training metrics

Data Pipeline:
    Database → Sampling → Augmentation → Batch Processing → Model Training
    
Usage Examples:
    # Basic training with config defaults
    python train_v2.py
    
    # ResNet with aggressive random sampling
    python train_v2.py --model-type resnet --sample-strategy random --sample-size 20000
    
    # Quick training run
    python train_v2.py --epochs 10 --batch-size 64

Output:
    - Updated model checkpoint
    - Training loss progression
    - Performance statistics
"""

import argparse
import os
import random
import torch
import numpy as np
from typing import Dict, Any, List

from game import get_all_orientations
from settings_loader import BOARD_SIZE, TABLE_NAME, DB_PATH
from db import Database
from config_loader import config


def create_model(model_type: str, board_size: int):
    """Create model instance from configuration."""
    ModelClass = config.get_model_class(model_type)
    return ModelClass(board_size)


def load_checkpoint(model, optimizer, checkpoint_path: str):
    """Load model and optimizer state from checkpoint."""
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        if optimizer and "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        print(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint.get("epoch", 0)
    return 0


def save_checkpoint(model, optimizer, checkpoint_path: str, epoch: int, losses: Dict[str, float]):
    """Save model and optimizer state to checkpoint."""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer else None,
        "epoch": epoch,
        "policy_loss": losses.get("policy_loss", 0),
        "value_loss": losses.get("value_loss", 0),
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch}")


def prepare_batch_data(training_data: List, db: Database, augmentation_config: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Prepare training batch from database records."""
    board_states = []
    player_captures_list = []
    opponent_captures_list = []
    policies = []
    values = []

    for data in training_data:
        id, board, player_captures, opponent_captures, num_moves, policy, value = db.decode_row(data)

        if augmentation_config.get("enabled", False):
            # Apply data augmentation with orientations
            num_orientations = augmentation_config.get("orientations", 8)
            
            # Get all possible orientations (up to 16)
            boards_orientations = get_all_orientations(board)
            policies_orientations = get_all_orientations(policy)
            
            # Limit to specified number of orientations
            boards_orientations = boards_orientations[:num_orientations]
            policies_orientations = policies_orientations[:num_orientations]
            
            orientation_pairs = list(zip(boards_orientations, policies_orientations))
        else:
            # No augmentation - just use original
            orientation_pairs = [(board, policy)]

        for board_oriented, policy_oriented in orientation_pairs:
            # Import here to avoid circular imports
            from models.gomoku_simple_nn import preprocess_game_state
            
            # Preprocess the game state
            board_tensor, player_cap_tensor, opponent_cap_tensor = preprocess_game_state(
                board_oriented, player_captures, opponent_captures
            )
            
            board_states.append(board_tensor)
            player_captures_list.append(player_cap_tensor)
            opponent_captures_list.append(opponent_cap_tensor)

            # Convert policy and value to tensors
            policy_tensor = torch.tensor(policy_oriented.copy(), dtype=torch.float32)
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

    return batch


def train_model(args):
    """Main training function with configuration support."""
    # Load training configuration
    train_config = config.get_training_config()
    
    # Override config with command line arguments
    model_type = args.model_type or train_config.get("model_type", "simple")
    batch_size = args.batch_size or train_config.get("batch_size", 32)
    learning_rate = args.learning_rate or train_config.get("learning_rate", 0.001)
    num_epochs = args.epochs or train_config.get("num_epochs", 50)
    data_sample_size = args.sample_size or train_config.get("data_sample_size", 10000)
    sample_strategy = args.sample_strategy or train_config.get("sample_strategy", "top_random")
    
    print(f"Training with model: {model_type}")
    print(f"Sample strategy: {sample_strategy}, Size: {data_sample_size}")
    print(f"Batch size: {batch_size}, Learning rate: {learning_rate}")
    print(f"Epochs: {num_epochs}")

    # Initialize database
    db = Database(DB_PATH, BOARD_SIZE)

    # Create model
    model = create_model(model_type, BOARD_SIZE[0])
    
    # Create optimizer
    weight_decay = train_config.get("weight_decay", 1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Load checkpoint
    checkpoint_dir = config.get("pipeline", "checkpoint_dir", "./checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, f"{model_type}_model.pt")
    start_epoch = load_checkpoint(model, optimizer, checkpoint_path)

    # Get augmentation config
    augmentation_config = train_config.get("augmentation", {"enabled": True, "orientations": 8})

    # Training loop
    save_interval = train_config.get("save_interval", 10)
    
    for epoch in range(start_epoch, start_epoch + num_epochs):
        print(f"\\n=== Epoch {epoch} ===")
        
        # Fetch training data with specified strategy
        training_data = db.fetch_collection(TABLE_NAME, data_sample_size, strategy=sample_strategy)

        if not training_data:
            print("No training data available")
            break

        print(f"Fetched {len(training_data)} records using '{sample_strategy}' strategy")

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

            # Prepare batch
            batch = prepare_batch_data(batch_data, db, augmentation_config)

            # Train on batch
            losses = model.train_on_batch(batch, epochs=1)

            # Accumulate losses
            epoch_policy_loss += losses["policy_loss"]
            epoch_value_loss += losses["value_loss"]
            num_batches += 1

            # Print progress
            if (batch_start // batch_size + 1) % 100 == 0:
                batch_num = batch_start // batch_size + 1
                total_batches = len(training_data) // batch_size
                print(f"  Batch {batch_num}/{total_batches}: "
                      f"Policy: {losses['policy_loss']:.4f}, Value: {losses['value_loss']:.4f}")

        # Calculate average losses
        avg_policy_loss = epoch_policy_loss / num_batches if num_batches > 0 else 0
        avg_value_loss = epoch_value_loss / num_batches if num_batches > 0 else 0

        print(f"Epoch {epoch} Results:")
        print(f"  Avg Policy Loss: {avg_policy_loss:.4f}")
        print(f"  Avg Value Loss: {avg_value_loss:.4f}")
        
        # Calculate total training samples (with augmentation)
        base_samples = len(training_data)
        if augmentation_config.get("enabled", False):
            augmentation_factor = augmentation_config.get("orientations", 8)
            total_samples = base_samples * augmentation_factor
        else:
            total_samples = base_samples
            
        print(f"  Total samples: {total_samples} (base: {base_samples})")

        # Save checkpoint
        if (epoch + 1) % save_interval == 0 or epoch == start_epoch + num_epochs - 1:
            losses_dict = {
                "policy_loss": avg_policy_loss,
                "value_loss": avg_value_loss
            }
            save_checkpoint(model, optimizer, checkpoint_path, epoch + 1, losses_dict)

    print(f"\\nTraining completed! Model saved to: {checkpoint_path}")


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description="AlphaPente Model Trainer v2")
    
    # Model configuration
    parser.add_argument("--model-type", type=str, choices=["simple", "resnet"], 
                       help="Model type to use (overrides config)")
    
    # Training parameters
    parser.add_argument("--batch-size", type=int, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    
    # Data sampling
    parser.add_argument("--sample-size", type=int, 
                       help="Number of records to sample from database")
    parser.add_argument("--sample-strategy", type=str, 
                       choices=["recent", "random", "top_random"],
                       help="Data sampling strategy")
    
    # Configuration
    parser.add_argument("--config", type=str, default="config/pipeline.json",
                       help="Path to configuration file")

    args = parser.parse_args()

    # Load configuration if different from default
    if args.config != "config/pipeline.json":
        global config
        from config_loader import ConfigLoader
        config = ConfigLoader(args.config)

    train_model(args)


if __name__ == "__main__":
    main()