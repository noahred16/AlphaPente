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
import numpy as np

# Initialize the database
db = Database(DB_PATH, BOARD_SIZE)

# Setup the database table for the game type
db.setup(TABLE_NAME)

# Initialize the model
model = GomokuSimpleNN(BOARD_SIZE[0])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
checkpoint_path = f"./checkpoints/{MODEL}.pt"

# Load checkpoint if it exists
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    print(f"Loaded checkpoint from {checkpoint_path}")

# Start the game
board, player_captures, opponent_captures = reset_game(BOARD_SIZE)
num_moves = 0

# record_limit = 10_000
record_limit = 1_000

record_count = 0

game_num = 0

mcts = MCTS(model, simulations=400, random=0.5)

# TODO Experiment with different MCTS configurations
p1 = MCTS(model, simulations=400)
p2 = MCTS(model, simulations=20)

p1 = mcts
p2 = mcts

while True:
    game_num += 1

    # start a game
    board, player_captures, opponent_captures = reset_game(BOARD_SIZE)
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
        move, policy = mcts.best_move(board, player_captures, opponent_captures)
        # moves.append move and policy
        moves.append(
            (
                np.array(board, copy=True),
                player_captures,
                opponent_captures,
                num_moves,
                policy,
            )
        )

        # pretty_print(board)
        # print(board)

        # make the move
        make_move(move, board)

        # flip board for display

        # print(
        #     f"Player {'1' if num_moves % 2 == 0 else '2'} Move {num_moves + 1}: {move}"
        # )
        num_moves += 1
        # pretty_print(board, move=move)
        # print(board)
        game_over = is_game_over(move, board, player_captures, CONNECT_N, num_moves)
        if game_over:
            result = calculate_score(num_moves, BOARD_SIZE, CAPTURES_ENABLED)
            # print(f"Game Over! Result: {result}")
        board = board * -1

    # store the moves in the db with the result
    # loop in reverse and flip the result
    for i, (board, player_captures, opponent_captures, num_moves, policy) in enumerate(
        reversed(moves)
    ):
        if record_count >= record_limit:
            break
        # flip the result for each opponent move
        result = result if i % 2 == 0 else -result
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

    print(
        f"Records in DB: {record_count}/{record_limit} - Game {game_num} completed in {len(moves)} moves."
    )

    if record_count >= record_limit:
        break
    # break
    # print()

    # from the moves, use the result and add to the db.


# generate until we hit 10 rows in the db using the model if it exists
