import pytest
import torch
import numpy as np
from models.gomoku_simple_nn import GomokuSimpleNN
from game import Game
from db import Database
from settings_loader import GameSettings

@pytest.fixture
def settings():
    return GameSettings("gomoku-simple")

@pytest.fixture
def game(settings):
    return Game(settings)

@pytest.fixture
def db(settings):
    return Database(settings.db_path, settings.board_size)

def test_gomoku_simple_nn():
    # Initialize the model
    board_size = (7, 7)
    model = GomokuSimpleNN(board_size)
    model.eval()

    # Create dummy inputs
    board = torch.zeros((1, *board_size), dtype=torch.float32)
    player_captures = torch.tensor(0, dtype=torch.float32)
    opponent_captures = torch.tensor(0, dtype=torch.float32)
    num_moves = torch.tensor(0, dtype=torch.float32)

    # Forward pass
    policy, value = model(board, player_captures, opponent_captures, num_moves)

    # Assertions
    assert policy.shape == (1, board_size[0] * board_size[1]), "Policy output shape is incorrect"
    assert value.shape == (1, 1), "Value output shape is incorrect"

def test_gomoku_simple_game_logic(game):
    # Test initial state
    assert game.board.shape == (7, 7), "Board size is incorrect"
    assert game.num_moves == 0, "Initial number of moves is incorrect"

    # Test legal moves
    legal_moves = game.get_legal_moves()
    assert len(legal_moves) == 1, "Initial legal moves should only include the center"
    assert legal_moves[0] == (3, 3), "Initial legal move should be the center"

    # Test making a move
    game.make_move((3, 3))
    assert game.board[3, 3] == 1, "Move was not applied correctly"
    assert game.num_moves == 1, "Number of moves did not update correctly"

def test_model_with_db(db, settings):
    # Fetch data from the database
    rows = db.fetch_collection(settings.table_name, 1)
    assert len(rows) > 0, "No data found in the database."

    # Decode the first row
    id, board, player_captures, opponent_captures, num_moves, policy, value = db.decode_row(rows[0])

    # Initialize the model
    model = GomokuSimpleNN(settings.board_size)
    model.eval()

    # Prepare inputs for the model
    board_tensor = torch.tensor(board, dtype=torch.float32).unsqueeze(0)
    player_captures_tensor = torch.tensor(player_captures, dtype=torch.float32)
    opponent_captures_tensor = torch.tensor(opponent_captures, dtype=torch.float32)
    num_moves_tensor = torch.tensor(num_moves, dtype=torch.float32)

    # Forward pass
    policy_pred, value_pred = model(board_tensor, player_captures_tensor, opponent_captures_tensor, num_moves_tensor)

    # Assertions for shape correctness
    assert policy_pred.shape == (1, settings.board_size[0] * settings.board_size[1]), "Policy output shape is incorrect."
    assert value_pred.shape == (1, 1), "Value output shape is incorrect."