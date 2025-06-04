import numpy as np


# this generic game class loads game settings and provides basic game functionality
# it does not keep track of the number of moves or whose turn it is. always assumes its player 1's turn.
# mostly works to enforce the rules: check for win and get legal moves.
class Game:
    def __init__(self, settings):
        self.settings = settings
        self.board_size = tuple(settings.board_size)
        self.captures_enabled = settings.captures_enabled
        self.tournament_rules_enabled = settings.tournament_rules_enabled
        self.connect_n = settings.connect_n
        self.reset_game()

    def reset_game(self):
        board = np.zeros(self.board_size, dtype=int)
        player_captures = 0
        opponent_captures = 0
        return board, player_captures, opponent_captures

    def get_legal_moves(self, board, num_moves):
        x_size, y_size = self.board_size
        if num_moves == 0:
            center = (x_size // 2, y_size // 2)
            return [center]

        legal_moves = []
        if num_moves == 2 and self.tournament_rules_enabled:
            # Ensure the move is at least 3 spaces away from the center
            center = (x_size // 2, y_size // 2)
            for i in range(x_size):
                for j in range(y_size):
                    if (abs(i - center[0]) >= 3 or abs(j - center[1]) >= 3) and board[
                        i
                    ][j] == 0:
                        legal_moves.append((i, j))
        else:
            for i in range(x_size):
                for j in range(y_size):
                    if board[i][j] == 0:
                        legal_moves.append((i, j))
        return legal_moves

    def make_move(self, move, board):
        x, y = move
        if board[x][y] != 0:
            raise ValueError(
                "Invalid move: Cell is already occupied. board: {}".format(board)
            )
        board[x][y] = 1

        return self.check_n_in_a_row(
            move, board
        )  # TODO: OR self.check_capture_limit(move, board)

        # max_num_moves = self.board_size[0] * self.board_size[1]
        # fastest_win = self.connect_n - 1

        # weighted_value = (max_num_moves - self.num_moves + fastest_win) / max_num_moves
        # scaled_value = weighted_value * 0.9 + 0.1

        # self.value = scaled_value
        # return self.value

    def check_n_in_a_row(self, move, board):
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        x, y = move
        for dx, dy in directions:
            count = 1
            nx, ny = x + dx, y + dy
            while (
                0 <= nx < self.board_size[0]
                and 0 <= ny < self.board_size[1]
                and board[nx][ny] == 1
            ):
                count += 1
                if count == self.connect_n:
                    return True
                nx += dx
                ny += dy
            nx, ny = x - dx, y - dy
            while (
                0 <= nx < self.board_size[0]
                and 0 <= ny < self.board_size[1]
                and board[nx][ny] == 1
            ):
                count += 1
                if count == self.connect_n:
                    return True
                nx -= dx
                ny -= dy
        return False


def pretty_print(board, move=None):
    board_size = board.shape[0]
    # Define symbols with similar-sized icons
    symbols = {
        0: "⚪",  # Empty (white circle)
        1: "🟢",  # Player 1 (green circle)
        -1: "🟤",  # Player 1 (blue circle)
    }

    if move is not None:
        x, y = move
        board[x][y] = 1

    # Print column headers with proper spacing
    print("   ", end="")
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
                print("🔴", end=" ")
            else:
                print(f"{symbols[piece]}", end=" ")
        print()
