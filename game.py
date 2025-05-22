import numpy as np


class Game:
    def __init__(self, settings):
        self.settings = settings
        self.board_size = tuple(settings.board_size)
        self.captures_enabled = settings.captures_enabled
        self.tournament_rules_enabled = settings.tournament_rules_enabled
        self.connect_n = settings.connect_n
        self.board = np.zeros(self.board_size, dtype=int)
        self.player_captures = 0
        self.opponent_captures = 0
        self.num_moves = 0
        self.policy = np.zeros(self.board_size, dtype=float)
        self.value = None

    def start_game(self):
        self.board = np.zeros(self.board_size, dtype=int)
        self.player_captures = 0
        self.opponent_captures = 0
        self.num_moves = 0
        self.policy = np.zeros(self.board_size, dtype=float)
        self.value = None

    def get_legal_moves(self):
        if self.num_moves == 0:
            center = (self.board_size[0] // 2, self.board_size[1] // 2)
            return [center]

        legal_moves = []
        if self.num_moves == 2 and self.tournament_rules_enabled:
            center = (self.board_size[0] // 2, self.board_size[1] // 2)
            min_x, max_x = center[0] - 3, center[0] + 3
            min_y, max_y = center[1] - 3, center[1] + 3
            for i in range(self.board_size[0]):
                for j in range(self.board_size[1]):
                    if (
                        not (min_x <= i <= max_x and min_y <= j <= max_y)
                        and self.board[i][j] == 0
                    ):
                        legal_moves.append((i, j))
        else:
            for i in range(self.board_size[0]):
                for j in range(self.board_size[1]):
                    if self.board[i][j] == 0:
                        legal_moves.append((i, j))
        return legal_moves

    def check_n_in_a_row(self, move):
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        x, y = move
        for dx, dy in directions:
            count = 1
            nx, ny = x + dx, y + dy
            while (
                0 <= nx < self.board_size[0]
                and 0 <= ny < self.board_size[1]
                and self.board[nx][ny] == 1
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
                and self.board[nx][ny] == 1
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
