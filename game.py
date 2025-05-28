import numpy as np


class Game:
    def __init__(self, settings):
        self.settings = settings
        self.board_size = tuple(settings.board_size)
        self.captures_enabled = settings.captures_enabled
        self.tournament_rules_enabled = settings.tournament_rules_enabled
        self.connect_n = settings.connect_n
        self.reset_game()

    def reset_game(self):
        self.board = np.zeros(self.board_size, dtype=int)
        self.player_captures = 0
        self.opponent_captures = 0
        self.num_moves = 0
        self.policy = np.zeros(self.board_size, dtype=float)
        self.value = None

    def flip_board(self):
        self.board = -self.board
        self.player_captures, self.opponent_captures = self.opponent_captures, self.player_captures

    def get_legal_moves(self):
        if self.num_moves == 0:
            center = (self.board_size[0] // 2, self.board_size[1] // 2)
            return [center]

        legal_moves = []
        if self.num_moves == 2 and self.tournament_rules_enabled:
            center = (self.board_size[0] // 2, self.board_size[1] // 2)
            for i in range(self.board_size[0]):
                for j in range(self.board_size[1]):
                    # Ensure the move is at least 3 spaces away from the center
                    if (
                        abs(i - center[0]) >= 3 or abs(j - center[1]) >= 3
                    ) and self.board[i][j] == 0:
                        legal_moves.append((i, j))
        else:
            for i in range(self.board_size[0]):
                for j in range(self.board_size[1]):
                    if self.board[i][j] == 0:
                        legal_moves.append((i, j))
        return legal_moves

    def make_move(self, move):
        x, y = move
        if self.board[x][y] != 0:
            raise ValueError("Invalid move: Cell is already occupied.")
        self.board[x][y] = 1
        self.num_moves += 1

        result = self.check_n_in_a_row(move)

        if result == False:
            return None

        max_num_moves = self.board_size[0] * self.board_size[1]
        fastest_win = self.connect_n - 1

        weighted_value = (max_num_moves - self.num_moves + fastest_win) / max_num_moves
        scaled_value = weighted_value * 0.9 + 0.1

        self.value = scaled_value

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
