from game import (
    reset_game,
    get_legal_moves,
    make_move,
    pretty_print,
    check_n_in_a_row,
    is_game_over,
    calculate_score,
)
from settings_loader import (
    BOARD_SIZE,
    TABLE_NAME,
    DB_PATH,
    TOURNAMENT_RULES_ENABLED,
    MODEL,
    CONNECT_N,
    CAPTURES_ENABLED,
)
from db import Database
from models.gomoku_simple_nn import GomokuSimpleNN
import random
import torch
import os
from mcts import MCTSNode, MCTS
import argparse
import numpy as np
import time



# plays a game given a model, returns moves. 
def play_game(model, simulations=100):
    # Track game play time
    game_start_time = time.time()

    # start a game
    board, player_captures, opponent_captures = reset_game(BOARD_SIZE)
    mcts = MCTS(model, simulations=simulations, random=0.3)

    # TODO Experiment with different MCTS configurations
    # p1 = MCTS(model, simulations=400)
    # p2 = MCTS(model, simulations=20)

    p1 = mcts
    p2 = mcts

    num_moves = 0
    moves = []
    move = None
    result = None

    # while game is not over make moves
    while result is None:
        if num_moves % 2 == 0:
            mcts = p1
        else:
            mcts = p2

        # generate the best move
        move, policy = mcts.best_move(
            board,
            player_captures,
            opponent_captures,
            starting_point=move,
            return_node=True,
        )
        # break
        # moves.append move and policy
        moves.append(
            (
                board.copy(),
                player_captures,
                opponent_captures,
                num_moves,
                policy,
            )
        )

        # pretty_print(board)
        # print(board)
        # print(move)
        # print(move.prev_move)

        move_to_make = move.prev_move
        make_move(move_to_make, board)
        num_moves += 1
        game_over = is_game_over(
            move_to_make, board, player_captures, CONNECT_N, num_moves
        )
        if game_over:
            result = calculate_score(num_moves, BOARD_SIZE, CAPTURES_ENABLED)
        board = board * -1

    game_end_time = time.time()
    game_duration = game_end_time - game_start_time

    p1.clear_tree()
    p2.clear_tree()

    return moves, result

def store_moves(moves, result):
    record_count = 0
    # Track DB store time
    db_start_time = time.time()
    # store the moves in the db with the result
    # loop in reverse and flip the result
    for i, (board, player_captures, opponent_captures, num_moves, policy) in enumerate(
        reversed(moves)
    ):
        # if record_count >= record_limit:
        #     break

        # skip over random moves (ones where policy is 0 for all
        if np.all(policy == 0):
            continue

        # flip the result for each opponent move
        result = result if i % 2 == 0 else -result
        # temp while speed check
        db.store(
            TABLE_NAME,
            board,
            player_captures,
            opponent_captures,
            num_moves,
            policy,
            result,
        )
        record_count += 1
    db_end_time = time.time()
    db_duration = db_end_time - db_start_time

    # print(
    #     f"Records in DB: {record_count}/{record_limit} - Game {game_num} completed in {len(moves)} moves."
    # )
    # print(f"Game {game_num} play time: {game_duration:.2f} seconds.")
    # print(f"Game {game_num} DB store time: {db_duration:.2f} seconds.")

    return record_count

    # break  # temp
# generate until we hit 10 rows in the db using the model if it exists



# if main
if __name__ == "__main__":
    


    # python generate.py --simulations 100
    parser = argparse.ArgumentParser(description="AlphaPente Data Generator")
    parser.add_argument(
        "--simulations",
        type=int,
        default=100,
        help="Number of MCTS simulations per move (default: 100)",
    )
    args = parser.parse_args()

    simulations = args.simulations
    print(
        f"Running AlphaPente data generation with {simulations} MCTS simulations per move."
    )

    # Initialize the database
    db = Database(DB_PATH, BOARD_SIZE)

    # Setup the database table for the game type
    db.setup(TABLE_NAME)

    # Initialize the model
    model = GomokuSimpleNN(BOARD_SIZE[0])
    checkpoint_path = f"./checkpoints/{MODEL}.pt"

    # Load checkpoint if it exists
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        print(f"Loaded checkpoint from {checkpoint_path}")

    # Start the game
    num_moves = 0

    # record_limit = 10_000
    record_limit = 1_000

    record_count = 0

    game_num = 0

    while True:
        game_num += 1

        print(f"Generate starting game")

        moves, result = play_game(model)
        print(f"Worker DONE game, with num_moves: {len(moves)}")

        # print("result", result)

        num_records = store_moves(moves, result)

        record_count += num_records

        if record_count >= record_limit:
            break




