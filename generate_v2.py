"""Enhanced Data Generation Script for AlphaPente Training Pipeline v2

This script generates training data by playing self-play games using MCTS and neural networks.
It's designed to be highly configurable and work as part of a coordinated training pipeline.

Key Improvements over v1:
- Configurable MCTS parameters (simulations, randomness, temperature)
- Support for multiple model types (simple CNN, ResNet)
- JSON configuration with command-line overrides
- Progress tracking and performance monitoring
- Structured logging and checkpointing
- Integration with pipeline coordination

How it works:
1. Loads model from checkpoint (or uses random weights if none exists)
2. Creates MCTS players with specified configuration
3. Plays self-play games, collecting (board_state, policy, result) tuples
4. Stores training data to SQLite database
5. Tracks progress and provides detailed statistics

Data Flow:
    Model → MCTS → Game Play → Policy Collection → Database Storage
    
Usage Examples:
    # Basic generation with config defaults
    python generate_v2.py
    
    # High-performance generation with ResNet
    python generate_v2.py --model-type resnet --simulations 500 --record-limit 5000
    
    # Custom MCTS parameters
    python generate_v2.py --mcts-random 0.1 --mcts-temperature 0.8

Output:
    - Training records stored in SQLite database
    - Progress logs with game statistics
    - Performance metrics (games/sec, records/sec)
"""

import argparse
import os
import time
import random
import torch
import numpy as np
from typing import Dict, Any, List, Tuple

from game import (
    reset_game, get_legal_moves, make_move, pretty_print, 
    check_n_in_a_row, is_game_over, calculate_score
)
from settings_loader import (
    BOARD_SIZE, TABLE_NAME, DB_PATH, TOURNAMENT_RULES_ENABLED,
    CONNECT_N, CAPTURES_ENABLED
)
from db import Database
from mcts import MCTS, MCTSNode
from config_loader import config


def create_model(model_type: str, board_size: int):
    """Create model instance from configuration."""
    ModelClass = config.get_model_class(model_type)
    return ModelClass(board_size)


def load_model_checkpoint(model, checkpoint_path: str):
    """Load model state from checkpoint."""
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        print(f"Loaded model checkpoint from {checkpoint_path}")
        return True
    else:
        print(f"No checkpoint found at {checkpoint_path}, using random weights")
        return False


def play_game(model, game_config: Dict[str, Any], game_id: int = 0) -> Tuple[List, float]:
    """
    Play a single game and return moves with policies.
    
    Args:
        model: Neural network model for MCTS
        game_config: Configuration for game generation
        game_id: Game identifier for logging
        
    Returns:
        Tuple of (moves_list, final_result)
    """
    start_time = time.time()
    
    # Initialize game
    board, player_captures, opponent_captures = reset_game(BOARD_SIZE)
    
    # Create MCTS players with configuration
    mcts_config = {
        "simulations": game_config.get("simulations", 100),
        "random": game_config.get("mcts_random", 0.3),
        "temperature": game_config.get("mcts_temperature", 1.0)
    }
    
    mcts = MCTS(model, **mcts_config)
    
    # For now, both players use same MCTS (self-play)
    p1 = mcts
    p2 = mcts
    
    num_moves = 0
    moves = []
    result = None

    # Game loop
    while result is None:
        current_mcts = p1 if num_moves % 2 == 0 else p2
        
        # Get best move
        move_node, policy = current_mcts.best_move(
            board, player_captures, opponent_captures,
            starting_point=None, return_node=True
        )
        
        # Store move data
        moves.append((
            board.copy(),
            player_captures,
            opponent_captures,
            num_moves,
            policy
        ))
        
        # Make the move
        move_to_make = move_node.prev_move
        make_move(move_to_make, board)
        num_moves += 1
        
        # Check for game over
        game_over = is_game_over(
            move_to_make, board, player_captures, CONNECT_N, num_moves
        )
        
        if game_over:
            result = calculate_score(num_moves, BOARD_SIZE, CAPTURES_ENABLED)
            
        # Flip board for next player
        board = board * -1

    # Cleanup
    p1.clear_tree()
    p2.clear_tree()
    
    game_duration = time.time() - start_time
    
    if game_id % 10 == 0:  # Log every 10 games
        print(f"  Game {game_id}: {len(moves)} moves, {game_duration:.2f}s, result: {result}")
    
    return moves, result


def store_game_moves(db: Database, moves: List, result: float, 
                    storage_config: Dict[str, Any]) -> int:
    """
    Store game moves to database.
    
    Args:
        db: Database instance
        moves: List of (board, captures, policy) tuples
        result: Game result value
        storage_config: Configuration for data storage
        
    Returns:
        Number of records stored
    """
    record_count = 0
    
    # Process moves in reverse order (from end game to start)
    for i, (board, player_captures, opponent_captures, num_moves, policy) in enumerate(reversed(moves)):
        # Skip moves with zero policy (shouldn't happen with proper MCTS)
        if np.all(policy == 0):
            continue
            
        # Flip result for alternating players
        flipped_result = result if i % 2 == 0 else -result
        
        # Store to database
        db.store(
            TABLE_NAME, board, player_captures, opponent_captures,
            num_moves, policy, flipped_result
        )
        record_count += 1
    
    return record_count


def generate_data(args):
    """Main data generation function."""
    # Load generation configuration
    gen_config = config.get_generation_config()
    
    # Override config with command line arguments
    model_type = args.model_type or gen_config.get("model_type", "simple")
    simulations = args.simulations or gen_config.get("simulations", 100)
    record_limit = args.record_limit or gen_config.get("record_limit", 1000)
    mcts_random = args.mcts_random if args.mcts_random is not None else gen_config.get("mcts_random", 0.3)
    mcts_temperature = args.mcts_temperature or gen_config.get("mcts_temperature", 1.0)
    
    print(f"Starting data generation with:")
    print(f"  Model type: {model_type}")
    print(f"  MCTS simulations: {simulations}")
    print(f"  MCTS random: {mcts_random}")
    print(f"  MCTS temperature: {mcts_temperature}")
    print(f"  Target records: {record_limit}")
    
    # Initialize database
    db = Database(DB_PATH, BOARD_SIZE)
    db.setup(TABLE_NAME)
    
    # Create and load model
    model = create_model(model_type, BOARD_SIZE[0])
    
    # Load checkpoint
    checkpoint_dir = config.get("pipeline", "checkpoint_dir", "./checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, f"{model_type}_model.pt")
    load_model_checkpoint(model, checkpoint_path)
    
    # Game configuration
    game_config = {
        "simulations": simulations,
        "mcts_random": mcts_random,
        "mcts_temperature": mcts_temperature
    }
    
    # Storage configuration
    storage_config = gen_config.get("storage", {})
    
    # Generation loop
    record_count = 0
    game_count = 0
    start_time = time.time()
    
    print(f"\\nGenerating games...")
    
    while record_count < record_limit:
        game_count += 1
        
        # Play game
        moves, result = play_game(model, game_config, game_count)
        
        # Store moves
        game_records = store_game_moves(db, moves, result, storage_config)
        record_count += game_records
        
        # Progress update
        progress = (record_count / record_limit) * 100
        print(f"Progress: {game_count} games, {record_count}/{record_limit} records ({progress:.1f}%)")
        
        # Save interval check
        save_interval = gen_config.get("save_interval_games", 50)
        if game_count % save_interval == 0:
            elapsed = time.time() - start_time
            rate = record_count / elapsed if elapsed > 0 else 0
            print(f"  Checkpoint: {elapsed:.1f}s elapsed, {rate:.1f} records/sec")
    
    # Final statistics
    total_time = time.time() - start_time
    avg_records_per_game = record_count / game_count if game_count > 0 else 0
    
    print(f"\\n=== Generation Complete ===")
    print(f"Total games: {game_count}")
    print(f"Total records: {record_count}")
    print(f"Avg records/game: {avg_records_per_game:.1f}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Records/sec: {record_count/total_time:.1f}")


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description="AlphaPente Data Generator v2")
    
    # Model configuration
    parser.add_argument("--model-type", type=str, choices=["simple", "resnet"],
                       help="Model type to use (overrides config)")
    
    # MCTS configuration
    parser.add_argument("--simulations", type=int,
                       help="MCTS simulations per move (overrides config)")
    parser.add_argument("--mcts-random", type=float,
                       help="MCTS random move probability (overrides config)")
    parser.add_argument("--mcts-temperature", type=float,
                       help="MCTS temperature for move selection (overrides config)")
    
    # Generation parameters
    parser.add_argument("--record-limit", type=int,
                       help="Target number of records to generate (overrides config)")
    
    # Configuration
    parser.add_argument("--config", type=str, default="config/pipeline.json",
                       help="Path to configuration file")
    
    # Debugging
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")

    args = parser.parse_args()

    # Load configuration if different from default
    if args.config != "config/pipeline.json":
        global config
        from config_loader import ConfigLoader
        config = ConfigLoader(args.config)

    # Set random seeds for reproducibility
    if hasattr(args, 'seed'):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    generate_data(args)


if __name__ == "__main__":
    main()