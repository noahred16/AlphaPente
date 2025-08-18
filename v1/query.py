from game import pretty_print
from db import Database
from settings_loader import (
    BOARD_SIZE,
    TABLE_NAME,
    DB_PATH,
)
import numpy as np

db = Database(DB_PATH, BOARD_SIZE)

# show_tables
tables = db.show_tables()
print(f"Tables in the database '{DB_PATH}':")
for table in tables:
    print(f" - {table}")

# get_count
count = db.get_count(TABLE_NAME)
print(f"\nCount of rows in table '{TABLE_NAME}': {count}")

# get top row from table: TABLE_NAME
# n = 1
# top_row = db.fetch_collection(TABLE_NAME, n)
# print(f"\nTop row {n} rows from table '{TABLE_NAME}':")
# for row in top_row:
#     id, board, player_captures, opponent_captures, num_moves, policy, value = (
#         db.decode_row(row)
#     )
#     print("id:", id)
#     # print("board:", board)
#     pretty_print(board)
#     print("player_captures:", player_captures)
#     print("opponent_captures:", opponent_captures)
#     print("num_moves:", num_moves)
#     # round policy 2-d numpy array to 2 decimal places for better readability
#     print("policy:", np.round(policy, 2))
#     print("value:", round(value, 2))
#     print("---------------------------------------------")
#     # print(row)


# review_last_game
last_game = db.review_last_game(TABLE_NAME)
for move in last_game:
    id, board, player_captures, opponent_captures, num_moves, policy, value = (
        db.decode_row(move)
    )
    print("id:", id)
    print("num_moves:", num_moves)
    odd = num_moves % 2 == 1
    pretty_print(board * (1 if odd else -1))
    # print("player_captures:", player_captures)
    # print("opponent_captures:", opponent_captures)
    # print("policy:", np.round(policy, 2))
    # print("value:", round(value, 2))
    print("---------------------------------------------")

print(f"\nReviewing last game from table '{TABLE_NAME}':")
print(f"Count of moves in last game: {len(last_game)}")

# python query.py --delete
# if script has the --delete flag, delete the table
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AlphaPente Query Script")
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete the table after querying (default: False)",
    )
    args = parser.parse_args()
    if args.delete:
        # clear_table
        db.clear_table(TABLE_NAME)
