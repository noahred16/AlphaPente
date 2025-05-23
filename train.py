from game import Game, pretty_print
from settings_loader import GameSettings
from db import Database
import random

# Load settings and initialize the game
settings = GameSettings("gomoku-simple")  # Change game type as needed
game = Game(settings)
db = Database(settings.db_path, settings.board_size)

# Setup the database table for the game type
db.setup(settings.table_name)

# Start the game
game.reset_game()

while True:
    legal_moves = game.get_legal_moves()

    if len(legal_moves) == 0:
        print("No legal moves available.")
        break

    move = random.choice(legal_moves)

    # Make a move using the make_move function
    game.make_move(move)

    if game.value is not None:
        db.store(
            settings.table_name,
            game.board,
            game.player_captures,
            game.opponent_captures,
            game.num_moves,
            game.policy,
            game.value,
        )

    if game.value is not None:
        print("Game Over with move:", move, "Number of moves:", game.num_moves)
        break

    # Flip the board for the opponent's turn
    game.board *= -1

# Pretty print the final board state
pretty_print(game.board, move)


# print how many records are in the database
total_records = db.get_total(settings.table_name)
print(f"Total records in {settings.table_name}: {total_records}")


rows = db.fetch_collection(settings.table_name, 50)

# for row in rows:
#     id, board, player_captures, opponent_captures, num_moves, policy, value = db.decode_row(row)
#     if value is not None:
#         # print(f"Number of moves: {num_moves}, Value: {value:.2f}")
#         print(f"ID: {id} Num Moves: {num_moves} Value: {value:.2f}")
