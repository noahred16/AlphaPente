from game import Game, pretty_print
from settings_loader import GameSettings
import random

# Load settings and initialize the game
settings = GameSettings("gomoku-simple")  # Change game type as needed
game = Game(settings)

# Start the game
game.start_game()

while True:
    legal_moves = game.get_legal_moves()

    if len(legal_moves) == 0:
        print("No legal moves available.")
        break

    move = random.choice(legal_moves)

    # Make a move
    x, y = move
    game.board[x][y] = 1  # Player 1's move
    result = game.check_n_in_a_row(move)

    if result:
        print("Game Over with move:", move, "Number of moves:", game.num_moves)
        break

    # Flip the board for the opponent's turn
    game.board *= -1
    game.num_moves += 1

# Pretty print the final board state
pretty_print(game.board, move)
