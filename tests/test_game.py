import pytest
from game import Game
from settings_loader import GameSettings


@pytest.mark.parametrize("game_type", ["gomoku", "gomoku-simple", "pente"])
def test_reset_game(game_type):
    game = Game(GameSettings(game_type))
    board, player_captures, opponent_captures = game.reset_game()
    assert board.shape == game.settings.board_size
    assert player_captures == 0
    assert opponent_captures == 0


@pytest.mark.parametrize("game_type", ["gomoku", "gomoku-simple", "pente"])
def test_get_legal_moves(game_type):
    game = Game(GameSettings(game_type))
    board, _player_captures, _opponent_captures = game.reset_game()
    legal_moves = game.get_legal_moves(board, num_moves=0)
    center = (game.settings.board_size[0] // 2, game.settings.board_size[1] // 2)
    assert len(legal_moves) == 1
    assert legal_moves[0] == center
