from game import reset_game, get_legal_moves, make_move
from settings_loader import BOARD_SIZE, TOURNAMENT_RULES_ENABLED


def test_reset_game():
    board, player_captures, opponent_captures = reset_game(BOARD_SIZE)
    assert board.shape == BOARD_SIZE
    assert player_captures == 0
    assert opponent_captures == 0


def test_get_legal_moves():
    board, _player_captures, _opponent_captures = reset_game(BOARD_SIZE)
    legal_moves = get_legal_moves(
        board, num_moves=0, tournament_rules_enabled=TOURNAMENT_RULES_ENABLED
    )
    center = (BOARD_SIZE[0] // 2, BOARD_SIZE[1] // 2)
    assert len(legal_moves) == 1
    assert legal_moves[0] == center
