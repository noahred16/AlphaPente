import pytest
import torch
import torch.nn as nn
import numpy as np
import random
from models.gomoku_simple_nn import GomokuSimpleNN, preprocess_game_state
from game import reset_game, get_legal_moves, make_move, pretty_print, is_game_over
from db import Database
from settings_loader import load_settings
from mcts import MCTSNode, MCTS
from tree_view import tree_visualization  # Import tree_visualization from tree_view.py
import os

_settings = load_settings("gomoku_simple")
BOARD_SIZE = _settings["board_size"]
CAPTURES_ENABLED = _settings["captures_enabled"]
TOURNAMENT_RULES_ENABLED = _settings["tournament_rules_enabled"]
CONNECT_N = _settings["connect_n"]
TABLE_NAME = _settings["table_name"]
DB_PATH = _settings["db_path"]
MODEL = _settings["model"]  # + "_TEST"


@pytest.fixture
def db():
    db = Database(DB_PATH, BOARD_SIZE)
    db.setup(TABLE_NAME)

    board, player_captures, opponent_captures = reset_game(BOARD_SIZE)
    num_moves = 9999
    policy = np.zeros((BOARD_SIZE[0] * BOARD_SIZE[1],), dtype=np.float32)
    value = 0.0
    db.store(
        TABLE_NAME,
        board,
        player_captures,
        opponent_captures,
        num_moves,
        policy,
        value,
    )

    yield db

    db.delete_by_num_moves(TABLE_NAME, 9999)


@pytest.fixture
def gomoku_model():
    # Initialize the model
    torch.manual_seed(42)
    model = GomokuSimpleNN(BOARD_SIZE[0])
    checkpoint_path = f"./checkpoints/{MODEL}.pt"

    # Load checkpoint if it exists
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"No checkpoint found at {checkpoint_path}. Using untrained model.")

    # seed and eval
    model.eval()

    return model


def test_gomoku_simple_nn(gomoku_model: GomokuSimpleNN):
    # Set the model to evaluation mode
    gomoku_model.eval()

    # Create an empty board (all zeros)
    empty_board = np.zeros((7, 7), dtype=np.int8)
    player_captures = 0
    opponent_captures = 0

    # Use the predict_policy_and_value function
    policy, value = gomoku_model.predict_policy_and_value(
        empty_board, player_captures, opponent_captures
    )

    # Test 1: Check policy shape
    assert policy.shape == (7, 7), f"Expected policy shape (7, 7), got {policy.shape}"

    # Test 2: Check that policy sums to 1 (within floating point tolerance)
    policy_sum = policy.sum()
    assert (
        abs(policy_sum - 1.0) < 1e-6
    ), f"Policy probabilities should sum to 1, got {policy_sum}"

    # Test 3: Check that all probabilities are non-negative
    assert (policy >= 0).all(), "All move probabilities should be non-negative"

    # Test 4: Check value range
    assert -1 <= value <= 1, f"Board value should be between -1 and 1, got {value}"

    # Test 5: For empty board, value should be close to 0 (neutral position)
    print(
        f"Empty board evaluation: {value:.3f} (should be close to 0 for a trained model)"
    )

    # Test 6: For empty board, policy should be relatively uniform
    # Calculate standard deviation of probabilities
    prob_std = policy.std()
    prob_mean = policy.mean()
    print(f"Policy statistics - Mean: {prob_mean:.4f}, Std: {prob_std:.4f}")
    print(f"(For empty board, expect relatively uniform distribution)")

    # Test 7: Print top 5 moves to verify they make sense
    flat_policy = policy.flatten()
    top5_probs_idx = np.argsort(flat_policy)[-5:][::-1]
    print("\nTop 5 predicted moves on empty board:")
    for i, idx in enumerate(top5_probs_idx):
        row, col = idx // 7, idx % 7
        prob = flat_policy[idx]
        print(f"  {i+1}. Position ({row}, {col}): {prob:.3f}")

    print("\nAll tests passed!")


def test_train_on_batch(gomoku_model: GomokuSimpleNN):
    """Test the train_on_batch function with a single sample."""

    # Store initial model state to verify weights change
    initial_params = {}
    for name, param in gomoku_model.named_parameters():
        initial_params[name] = param.clone().detach()

    # Create a single training sample
    batch_size = 1

    # Empty board state
    empty_board = np.zeros((7, 7), dtype=np.int8)
    player_captures = 0
    opponent_captures = 0

    # Preprocess the board state
    board_tensor, player_cap_tensor, opponent_cap_tensor = preprocess_game_state(
        empty_board, player_captures, opponent_captures
    )

    # Create fake target policy - let's say we want the model to play in the center
    target_policy = torch.zeros(49)
    center_position = 3 * 7 + 3  # (3, 3) in flattened coordinates
    target_policy[center_position] = 1.0  # One-hot encoding for center move

    # Create fake target value - let's say the position is slightly favorable
    target_value = torch.tensor([0.3])  # Slight advantage for current player

    # Prepare batch
    training_batch = {
        "states": board_tensor.unsqueeze(0),  # Add batch dimension
        "player_captures": player_cap_tensor.unsqueeze(0),  # Add batch dimension
        "opponent_captures": opponent_cap_tensor.unsqueeze(0),  # Add batch dimension
        "policies": target_policy.unsqueeze(0),  # Add batch dimension
        "values": target_value,
    }

    # Test 1: Verify batch shapes
    assert training_batch["states"].shape == (
        1,
        2,
        7,
        7,
    ), f"Wrong states shape: {training_batch['states'].shape}"
    assert training_batch["player_captures"].shape == (
        1,
    ), f"Wrong player_captures shape: {training_batch['player_captures'].shape}"
    assert training_batch["opponent_captures"].shape == (
        1,
    ), f"Wrong opponent_captures shape: {training_batch['opponent_captures'].shape}"
    assert training_batch["policies"].shape == (
        1,
        49,
    ), f"Wrong policies shape: {training_batch['policies'].shape}"
    assert training_batch["values"].shape == (
        1,
    ), f"Wrong values shape: {training_batch['values'].shape}"
    print("✓ Batch shapes are correct")

    # Get predictions before training
    pre_train_policy, pre_train_value = gomoku_model.predict_policy_and_value(
        empty_board, player_captures, opponent_captures
    )

    # Train on the batch
    losses = gomoku_model.train_on_batch(training_batch, epochs=10)

    # Test 2: Verify losses are returned and are reasonable
    assert "policy_loss" in losses, "Missing policy_loss in returned losses"
    assert "value_loss" in losses, "Missing value_loss in returned losses"
    assert "total_loss" in losses, "Missing total_loss in returned losses"
    assert (
        losses["policy_loss"] >= 0
    ), f"Policy loss should be non-negative, got {losses['policy_loss']}"
    assert (
        losses["value_loss"] >= 0
    ), f"Value loss should be non-negative, got {losses['value_loss']}"
    assert (
        abs(losses["total_loss"] - (losses["policy_loss"] + losses["value_loss"]))
        < 1e-6
    ), "Total loss should equal sum of policy and value losses"
    print(
        f"✓ Losses computed correctly - Policy: {losses['policy_loss']:.4f}, Value: {losses['value_loss']:.4f}"
    )


# def test_train_model_with_db(db: Database, gomoku_model: GomokuSimpleNN):
#     # Fetch data from the database
#     rows = db.fetch_collection(TABLE_NAME, 1)
#     assert len(rows) == 1, f"Unexpected number of rows fetched: {len(rows)}"

#     # Prepare inputs for the model. Decode and convert to tensors
#     batch = []
#     for row in rows:
#         id, board, player_captures, opponent_captures, num_moves, policy, value = db.decode_row(row)

#     # TODO test the model training with the fetched data
#     # batch will just be of size 1 but its just a to test input shape.


def test_train_model_with_db(db: Database, gomoku_model: GomokuSimpleNN):
    # Fetch data from the database
    rows = db.fetch_collection(TABLE_NAME, 1)
    assert len(rows) == 1, f"Unexpected number of rows fetched: {len(rows)}"

    # Prepare inputs for the model. Decode and convert to tensors
    batch = []
    for row in rows:
        id, board, player_captures, opponent_captures, num_moves, policy, value = (
            db.decode_row(row)
        )
        batch.append(
            {
                "board": board,
                "player_captures": player_captures,
                "opponent_captures": opponent_captures,
                "policy": policy,
                "value": value,
            }
        )
        # TODO convert to tensors first

    # TODO test the model train_on_batch with the fetched data


def test_gomoku_simple_game_logic():
    # Test initial state
    board, player_captures, opponent_captures = reset_game(BOARD_SIZE)
    assert board.shape == BOARD_SIZE, "Board size is incorrect"
    assert player_captures == 0, "Initial player captures should be zero"
    assert opponent_captures == 0, "Initial opponent captures should be zero"

    # Test legal moves
    legal_moves = get_legal_moves(
        board, num_moves=0, tournament_rules_enabled=TOURNAMENT_RULES_ENABLED
    )
    assert len(legal_moves) == 1, "Initial legal moves should only include the center"
    assert legal_moves[0] == (
        BOARD_SIZE[0] // 2,
        BOARD_SIZE[1] // 2,
    ), "Initial legal move should be the center"

    # Test making a move
    move = (3, 3)
    make_move(move, board)
    assert board[move] == 1, "Move was not applied correctly"

    # flip board and make move another move
    board *= -1
    player_captures, opponent_captures = player_captures, opponent_captures

    another_move = (3, 4)
    make_move((3, 4), board)
    assert (
        board[another_move] == 1
    ), "Move was not applied correctly after flipping board"
    assert (
        board[move] == -1
    ), "Previous move should remain unchanged after flipping board"


def test_mcts0(gomoku_model: GomokuSimpleNN):
    board, player_captures, opponent_captures = reset_game(BOARD_SIZE)
    board = np.array(
        [
            ["O", "X", "O", "X", "O", "X", "O"],
            ["X", "O", "X", "X", "X", "O", "X"],
            ["X", "O", "X", "O", "X", "O", "O"],
            ["O", "X", "O", "O", "O", "O", "O"],
            ["O", "X", "X", "O", "O", "X", "O"],
            ["X", "X", "X", "X", "O", "X", "X"],
            ["O", "O", " ", " ", " ", "X", "X"],
        ]
    )

    num_of_x = np.count_nonzero(board == "X")
    num_of_o = np.count_nonzero(board == "O")

    assert num_of_x == 23
    assert num_of_o == 23

    # convert X to 1
    # convert O to -1

    board = np.where(board == "X", 1, board)
    board = np.where(board == "O", -1, board)
    board = np.where(board == " ", 0, board)
    board = board.astype(np.int8)

    model = gomoku_model
    mcts = MCTS(model, simulations=800)

    root = mcts.run(board, player_captures, opponent_captures)
    assert root is not None, "MCTS run should return a valid result"

    # Generate tree visualization for debugging
    tree_visualization(root)

    best_child = root.best_child(greedy=True)
    move = best_child.prev_move

    # print("Best move is", move)
    # pretty_print(board, move)
    # print("Best move is", move)
    # print("")

    assert move is not None
    # best move is (5,3)
    assert move == (6, 4), f"Expected best move to be (6, 4), got {move}"


# command to run this test
# pytest tests/test_gomoku_simple.py::test_mcts1
def test_mcts1(gomoku_model: GomokuSimpleNN):
    board, player_captures, opponent_captures = reset_game(BOARD_SIZE)
    # board = np.array([
    #     [0, 0, 0, 0, 0,  0, 0],
    #     [0, 0, 0, 0, 0,  0, 0],
    #     [0, 0, 0, 1, 0, -1, 0],
    #     [0, 0, 0, 1, 0, -1, 0],
    #     [0, 0, 0, 1, 0, -1, 0],
    #     [0, 0, 0, 1, 0, -1, 0],
    #     [0, 0, 0, 0, 0,  0, 0],
    # ])

    # board = np.array([
    #     [0, 0, 0, 0, 0,  0, 0],
    #     [0, 0, 0, 0, 0,  0, 0],
    #     [0, 0, 0, 1, 0, -1, 0],
    #     [0, 0, 0, 1, 0, -1, 0],
    #     [0, 0, 0, 1, 0, -1, 0],
    #     [0, 0, 0, 0, 0,  0, 0],
    #     [0, 0, 0, 0, 0,  0, 0],
    # ])

    # board = np.array([
    #     [0, 0, 0, 0, 0,  0, 0],
    #     [0, 0, 0, 0, 0,  0, 0],
    #     [0, 0, 0, 1, 0, -1, 0],
    #     [0, 0, 1, 0, 0, -1, 0],
    #     [0, 0, 0, 1, 0, -1, 0],
    #     [0, 0, 0, 0, 0,  0, 0],
    #     [0, 0, 0, 0, 0,  0, 0],
    # ])

    # board = np.array(
    #     [
    #         ["O", "X", "O", "X", "O", "X", "O"],
    #         ["X", "O", "X", "O", "X", "O", "X"],
    #         ["X", "O", "X", "O", "X", "O", "X"],
    #         ["O", "X", "O", "X", "O", "X", "O"],
    #         ["X", "O", "X", "O", "X", "O", "X"],
    #         ["O", "X", "O", "X", "O", "O", "X"],
    #         ["O", "X", "O", " ", " ", " ", " "],
    #     ]
    # )

    # 5 moves
    # 1 leads to a win in 2. 1! yay
    # 2 leads to a loss in 3. -1 :(
    # 2 lead to a tie.  0 okay
    board = np.array(
        [
            ["O", "X", "O", "X", "O", "X", "O"],
            ["X", "O", "X", "X", "X", "O", "X"],
            ["X", "O", "X", "O", "X", "O", "O"],
            ["O", "X", "O", "O", "O", "O", "O"],
            ["O", "X", "X", "O", "O", "X", "O"],
            ["X", "X", "X", " ", " ", "X", "X"],
            ["O", "O", " ", " ", " ", "X", "X"],
        ]
    )

    num_of_x = np.count_nonzero(board == "X")
    num_of_o = np.count_nonzero(board == "O")

    assert num_of_x == 22
    assert num_of_o == 22

    # convert X to 1
    # convert O to -1

    board = np.where(board == "X", 1, board)
    board = np.where(board == "O", -1, board)
    board = np.where(board == " ", 0, board)
    board = board.astype(np.int8)

    model = gomoku_model
    mcts = MCTS(model, simulations=1800)

    root = mcts.run(board, player_captures, opponent_captures)
    assert root is not None, "MCTS run should return a valid result"

    # Generate tree visualization for debugging
    tree_visualization(root)

    best_child = root.best_child(greedy=True)
    move = best_child.prev_move

    # print("Best move is", move)
    # pretty_print(board, move)
    # print("Best move is", move)
    # print("")

    assert move is not None
    # best move is (5,3)
    assert move == (5, 3), f"Expected best move to be (5, 3), got {move}"

    # children = root.children

    # # print children scores
    # for move, child in children.items():
    #     print(
    #         f"  Child move: {move}, value: {child.value}, num_visits: {child.num_visits}, is_terminal: {child.is_terminal}, num_children: {len(child.children)}, is_fully_expanded: {child.is_fully_expanded()}, puct: {child.puct}"
    #     )
    #     pretty_print(child.board, child.prev_move)
    #     next_level_children = child.children
    #     for next_level_move, next_level_child in next_level_children.items():
    #         print(
    #             f"    Next level child move: {next_level_move}, value: {next_level_child.value}, num_visits: {next_level_child.num_visits}, is_terminal: {next_level_child.is_terminal}, num_children: {len(next_level_child.children)}, puct: {next_level_child.puct}"
    #         )
    #         pretty_print(next_level_child.board, next_level_child.prev_move)
    #     print("---")
    #     print("---")
    #     print("---")
    # print(child)

    # seems to never consider ties...
    # 2 lead to a tie


# pytest tests/test_gomoku_simple.py::test_mcts1
def test_mcts2(gomoku_model: GomokuSimpleNN):
    board, player_captures, opponent_captures = reset_game(BOARD_SIZE)
    # 3 moves
    # 1 leads to insta win
    # 2 lead to a tie
    board = np.array(
        [
            ["O", "X", "O", "X", "X", "X", "O"],
            ["X", "O", "X", "X", "X", "O", "X"],
            ["X", "O", "X", "O", "O", "O", "O"],
            ["O", "X", "O", "O", "O", "O", "O"],
            ["O", "X", "X", "O", "O", "X", "O"],
            ["X", "X", "X", "X", " ", "X", "X"],
            ["O", "O", " ", " ", "O", "X", "X"],
        ]
    )

    num_of_x = np.count_nonzero(board == "X")
    num_of_o = np.count_nonzero(board == "O")

    assert num_of_x == 23
    assert num_of_o == 23

    # convert X to 1
    # convert O to -1

    board = np.where(board == "X", 1, board)
    board = np.where(board == "O", -1, board)
    board = np.where(board == " ", 0, board)
    board = board.astype(np.int8)

    model = gomoku_model
    mcts = MCTS(model, simulations=200)

    root = mcts.run(board, player_captures, opponent_captures)
    assert root is not None, "MCTS run should return a valid result"

    # Generate tree visualization for debugging
    # tree_visualization(root)

    best_child = root.best_child(greedy=True)
    move = best_child.prev_move

    move, policy = mcts.best_move(board, player_captures, opponent_captures)

    print("Best move is", move)
    # print("board", board)
    pretty_print(board, move)
    print("Best move is", move)
    print("")

    assert move is not None
    # best move is (5,3)
    assert move == (5, 4), f"Expected best move to be (5,4), got {move}"


def test_mcts3(gomoku_model: GomokuSimpleNN):
    board, player_captures, opponent_captures = reset_game(BOARD_SIZE)

    # 5 moves
    # 1 leads to a win in 2. 1! yay
    # 2 leads to a loss in 3. -1 :(
    # 2 lead to a tie.  0 okay
    board = np.array(
        [
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", "X", " ", "O", " "],
            [" ", " ", " ", "X", " ", "O", " "],
            [" ", " ", " ", "X", " ", "O", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
        ]
    )

    num_of_x = np.count_nonzero(board == "X")
    num_of_o = np.count_nonzero(board == "O")

    assert num_of_x == 3
    assert num_of_o == 3

    # convert X to 1
    # convert O to -1

    board = np.where(board == "X", 1, board)
    board = np.where(board == "O", -1, board)
    board = np.where(board == " ", 0, board)
    board = board.astype(np.int8)

    model = gomoku_model
    mcts = MCTS(model, simulations=200)

    root = mcts.run(board, player_captures, opponent_captures)
    assert root is not None, "MCTS run should return a valid result"

    # Generate tree visualization for debugging
    # tree_visualization(root)

    # best_child = root.best_child(greedy=True)
    # move = best_child.prev_move

    move, policy = mcts.best_move(board, player_captures, opponent_captures)

    print("Best move is", move)
    pretty_print(board, move)
    print("Best move is", move)
    print("")

    assert move is not None
    # move is either (1, 3) or (6, 3)
    assert move in [
        (1, 3),
        (5, 3),
    ], f"Expected best move to be either (1, 3) or (5, 3), got {move}"


def test_cases_mcts(gomoku_model: GomokuSimpleNN):
    cases = [
        {
            "board": np.array(
                [
                    [" ", " ", " ", " ", " ", " ", " "],
                    [" ", " ", " ", " ", " ", " ", " "],
                    [" ", " ", " ", "X", " ", "O", " "],
                    [" ", " ", " ", "X", " ", "O", " "],
                    [" ", " ", " ", "X", " ", "O", " "],
                    [" ", " ", " ", " ", " ", " ", " "],
                    [" ", " ", " ", " ", " ", " ", " "],
                ]
            ),
            "expected_moves": [(1, 3), (5, 3)],
        },
    ]

    model = gomoku_model

    for case in cases:
        board, player_captures, opponent_captures = reset_game(BOARD_SIZE)
        board = case["board"]
        expected_moves = case["expected_moves"]

        board = np.where(board == "X", 1, board)
        board = np.where(board == "O", -1, board)
        board = np.where(board == " ", 0, board)
        board = board.astype(np.int8)

        mcts = MCTS(model, simulations=200)
        move, policy = mcts.best_move(board, player_captures, opponent_captures)

        assert move is not None, "MCTS should return a valid move"
        assert (
            move in expected_moves
        ), f"Expected move to be one of {expected_moves}, got {move}"

        # tree_visualization(root)
        # pretty_print(board, move)


# TODO write a test called test_mcts4


# [[ 0  0  1  0  0  0  0]
#  [ 0  0  0  0  0  0  0]
#  [ 0  0  0  0  0  0  0]
#  [ 0  0 -1 -1 -1  0  0]
#  [ 0  0  0  0  0  0  0]
#  [ 0  0  0  1  1  0  0]
#  [-1  0  0  0  0  0  0]]
def test_mcts4(gomoku_model: GomokuSimpleNN):
    """Test MCTS with a specific board position requiring strategic play."""
    board, player_captures, opponent_captures = reset_game(BOARD_SIZE)

    # Board position from the comment at end of file
    board = np.array(
        [
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, -1, -1, -1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0],
            [-1, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.int8,
    )

    # Analyze the position:
    # Player 1 (X) has: stones at (0,2), (5,3), (5,4)
    # Player -1 (O) has: stones at (3,2), (3,3), (3,4), (6,0)
    # O has 3 in a row horizontally at row 3
    # X has 2 in a row horizontally at row 5

    # Critical moves to consider:
    # - (3,1) or (3,5) to block O from getting 4 in a row
    # - (5,2) or (5,5) to extend X's line to 3

    # Since it's X's turn and O has 3 in a row, X must block!
    expected_blocking_moves = [(3, 1), (3, 5)]

    model = gomoku_model
    mcts = MCTS(model, simulations=300)

    root = mcts.run(board, player_captures, opponent_captures)
    assert root is not None, "MCTS run should return a valid result"

    # Get the best move
    move, policy = mcts.best_move(board, player_captures, opponent_captures)

    # tree_visualization(root)

    print(f"\ntest_mcts4 - Board position:")
    # pretty_print(board, move)
    print(board)
    print(policy)
    print(f"Best move selected: {move}")
    print(f"Expected blocking moves: {expected_blocking_moves}")

    # Verify that MCTS chooses to block O's potential win
    assert move is not None, "MCTS should return a valid move"
    assert move in expected_blocking_moves, (
        f"Expected MCTS to block O's line by playing one of {expected_blocking_moves}, "
        f"but it played {move} instead"
    )

    # Additional validation: check that the move is actually blocking
    # if move == (3, 1):
    #     print("✓ Correctly blocked on the left side of O's line")
    # elif move == (3, 5):
    #     print("✓ Correctly blocked on the right side of O's line")

    # Optional: Analyze top moves from MCTS
    # print("\nTop moves considered by MCTS:")
    # children = root.children
    # sorted_children = sorted(
    #     children.items(),
    #     key=lambda x: x[1].num_visits,
    #     reverse=True
    # )[:5]

    # for i, (child_move, child_node) in enumerate(sorted_children):
    #     visit_ratio = child_node.num_visits / root.num_visits
    #     avg_value = child_node.value / child_node.num_visits if child_node.num_visits > 0 else 0
    #     print(f"  {i+1}. Move {child_move}: "
    #           f"visits={child_node.num_visits} ({visit_ratio:.1%}), "
    #           f"avg_value={avg_value:.3f}")

    # Uncomment to generate tree visualization for debugging
    # tree_visualization(root)


# test set to ensure that the model is learning.
# any open 3 should be blocked.
# generate 100 test cases with open 3s
# and ensure that the model blocks them.
def test_mcts_open_3_blocking(gomoku_model: GomokuSimpleNN):
    """Test MCTS with multiple scenarios where it should block open 3s."""
    model = gomoku_model
    mcts = MCTS(model, simulations=300)

    # np seed
    np.random.seed(42)
    random.seed(42)

    opp = -1
    user = 1

    total = 1
    cases = []

    # Generate 100 test cases with open 3s
    for i in range(total):
        board = np.zeros((7, 7), dtype=np.int8)
        # chose any point between (2,2) and (4,4) as the center of the open 3
        center_row = np.random.randint(2, 5)
        center_col = np.random.randint(2, 5)

        # place 3 in a row horizontally
        board[center_row, center_col - 1] = opp
        board[center_row, center_col] = opp
        board[center_row, center_col + 1] = opp

        # place the user at the center if its not taken (3,3)
        if board[3, 3] == 0:
            board[3, 3] = user

        blocking_moves = [
            (center_row, center_col - 2),  # Block left side
            (center_row, center_col + 2),  # Block right side
        ]

        # place 2 user stones randomly on the board where there is an empty space but not in a blocked position
        for _ in range(2):
            while True:
                row = np.random.randint(0, 7)
                col = np.random.randint(0, 7)
                if board[row, col] == 0 and (row, col) not in blocking_moves:
                    board[row, col] = user
                    break

        # print(f"Test case {i+1} - Board position:")
        # pretty_print(board)

        cases.append(
            {
                "board": board,
                "blocking_moves": blocking_moves,
            }
        )

    # TODO use batch predictions to see if the AI learned how to block yet...
    # assert 100 cases
    # assert len(cases) == 100, f"Expected 100 test cases, got {len(cases)}"

    # # Run MCTS to find the best move
    # root = mcts.run(board, 0, 0)
    # assert root is not None, "MCTS run should return a valid result"

    # move, policy = mcts.best_move(board, 0, 0)
    # print(f"\nTest case {i+1} - Best move selected: {move}")

    # # Verify that MCTS chooses to block the open 3
    # assert move is not None, "MCTS should return a valid move"
    # assert move in blocking_moves, (
    #     f"Expected MCTS to block open 3 by playing one of {blocking_moves}, "
    #     f"but it played {move} instead"
    # )

    # Example usage of the new batch prediction method:
    # Create model

    # Create sample batch of game states
    # game_states = []
    # for i in range(5):  # 5 different game states
    #     # Random board state for example
    #     board = np.random.choice([0, 1, -1], size=(7, 7), p=[0.7, 0.15, 0.15])
    #     player_captures = np.random.randint(0, 5)
    #     opponent_captures = np.random.randint(0, 5)
    #     game_states.append((board, player_captures, opponent_captures))

    # # Predict for entire batch
    # policies, values = model.predict_batch(game_states)

    # print(f"Predicted {len(policies)} policies and {len(values)} values")
    # print(f"First policy shape: {policies[0].shape}")
    # print(f"First value: {values[0]:.4f}")

    game_states = []
    for case in cases:
        board = case["board"]
        player_captures = 0
        opponent_captures = 0
        game_states.append((board, player_captures, opponent_captures))

    # Predict for entire batch
    policies, values = model.predict_batch(game_states)
    print(f"Predicted {len(policies)} policies and {len(values)} values")

    num_correct = 0

    # Check if the model is blocking the open 3s
    for i, case in enumerate(cases):
        blocking_moves = case["blocking_moves"]
        policy = policies[i]
        value = values[i]

        # Get the top move from the policy
        top_move_idx = np.argmax(policy)
        top_move_row = top_move_idx // 7
        top_move_col = top_move_idx % 7
        top_move = (top_move_row, top_move_col)

        print(f"Test case {i+1} - Best move selected: {top_move}, Value: {value:.4f}")
        print(f"Expected blocking moves: {blocking_moves}")
        print(f"Policy:\n {np.round(policy.reshape(7, 7), 2)}")

        if top_move in blocking_moves:
            num_correct += 1
    print(f"Number of correct blocking moves: {num_correct} out of {len(cases)}")

    assert (
        num_correct >= 80
    ), f"Expected at least 80% correct blocking moves, got {num_correct} out of {len(cases)}"
