import numpy as np


def reset_game(board_size):
    board = np.zeros(board_size, dtype=int)
    player_captures = 0
    opponent_captures = 0
    return board, player_captures, opponent_captures


def get_legal_moves(board, num_moves, tournament_rules_enabled):
    x_size, y_size = board.shape
    if num_moves == 0:
        center = (x_size // 2, y_size // 2)
        return [center]

    legal_moves = []
    if num_moves == 2 and tournament_rules_enabled:
        # Ensure the move is at least 3 spaces away from the center
        center = (x_size // 2, y_size // 2)
        for i in range(x_size):
            for j in range(y_size):
                if (abs(i - center[0]) >= 3 or abs(j - center[1]) >= 3) and board[i][
                    j
                ] == 0:
                    legal_moves.append((i, j))
    else:
        for i in range(x_size):
            for j in range(y_size):
                if board[i][j] == 0:
                    legal_moves.append((i, j))
    return legal_moves


def make_move(move, board, player=1):
    x, y = move
    if board[x][y] != 0:
        raise ValueError(
            "Invalid move: Cell is already occupied. board: {}".format(board)
        )
    board[x][y] = player


def is_game_over(move, board, player_captures, connect_n, num_moves):
    size = board.shape[0]
    if move is None:
        return False
    result = check_n_in_a_row(move, board, connect_n) or player_captures >= 5
    if result:
        return True
    if num_moves == size * size:
        return True
    return False


def calculate_score(num_moves, board_size, captures_enabled):
    size, _ = board_size
    if num_moves == size * size:  # tie
        return 0

    worst_case_score = (
        size * size
    )  # technically w captures, it could get worse, but like dang no way right?
    best_case_score = 5 * 2 - 1

    scaled_score = (num_moves - best_case_score) / (worst_case_score - best_case_score)

    score = 1 - scaled_score  # Invert the score to make fewer moves better

    # linear transformation [0.1, 1]
    score = 0.1 + (score * (1 - 0.1))

    return score


def check_n_in_a_row(move, board, connect_n, current_player=1):
    board_size = board.shape
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    x, y = move
    for dx, dy in directions:
        count = 1
        nx, ny = x + dx, y + dy
        while (
            0 <= nx < board_size[0]
            and 0 <= ny < board_size[1]
            and board[nx][ny] == current_player
        ):
            count += 1
            if count == connect_n:
                return True
            nx += dx
            ny += dy
        nx, ny = x - dx, y - dy
        while (
            0 <= nx < board_size[0]
            and 0 <= ny < board_size[1]
            and board[nx][ny] == current_player
        ):
            count += 1
            if count == connect_n:
                return True
            nx -= dx
            ny -= dy
    return False


def pretty_print(board_ref, move=None, ASCII=False):
    board = np.array(board_ref, copy=True)
    board_size = board.shape[0]
    # Define symbols with similar-sized icons
    symbols = {
        0: "⚪",  # Empty (white circle)
        1: "🟢",  # Player 1 (green circle)
        -1: "🟤",  # Player 1 (blue circle)
    }
    final_move = "🔴"
    if ASCII:
        symbols = {
            0: " ",  # Empty
            1: "X",  # Player 1
            -1: "O",  # Player 2
        }
        final_move = "*"

    if move is not None:
        x, y = move
        board[x][y] = 1

    # Print column headers with proper spacing
    print("     ", end="")
    for i in range(board_size):
        print(f"{i:2d}", end=" ")
    print()

    # Print horizontal line
    print("   " + "═══" * board_size)

    # Print rows with row numbers and symbols
    for i in range(board_size):
        print(f"{i:2d} ║", end=" ")
        for j in range(board_size):
            piece = int(board[i][j])
            if move is not None and (i, j) == move:
                print(final_move, end=" ")
            else:
                print(f"{symbols[piece]}", end=" ")
        print()


def get_all_orientations(board):
    """
    Generate all 16 possible orientations of a board through rotations and flips.

    The 16 orientations come from:
    - 4 rotations (0°, 90°, 180°, 270°)
    - 2 states for each rotation (original and horizontally flipped)
    - 2 states for each of those (original and vertically flipped)

    However, due to symmetries, we actually get 8 unique transformations:
    - 4 rotations
    - 4 rotations of the horizontally flipped board

    Args:
        board: numpy array representing the board

    Returns:
        list of 16 numpy arrays representing all possible orientations
    """
    orientations = []

    # Original board - 4 rotations
    for k in range(4):
        orientations.append(np.rot90(board, k))

    # Horizontally flipped board - 4 rotations
    h_flipped = np.fliplr(board)
    for k in range(4):
        orientations.append(np.rot90(h_flipped, k))

    # Vertically flipped board - 4 rotations
    v_flipped = np.flipud(board)
    for k in range(4):
        orientations.append(np.rot90(v_flipped, k))

    # Both horizontally and vertically flipped board - 4 rotations
    hv_flipped = np.fliplr(np.flipud(board))
    for k in range(4):
        orientations.append(np.rot90(hv_flipped, k))

    return orientations
