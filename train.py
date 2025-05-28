from game import Game, pretty_print
from settings_loader import GameSettings
from db import Database
from models.gomoku_simple_nn import GomokuSimpleNN
import random
import torch
import os

# Load settings and initialize the game
settings = GameSettings("gomoku-simple")  # Change game type as needed
game = Game(settings)
db = Database(settings.db_path, settings.board_size)

# Setup the database table for the game type
db.setup(settings.table_name)

# Initialize the model
model = GomokuSimpleNN(settings.board_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
checkpoint_path = f"./checkpoints/{settings.settings['model']}.pt"

# Load checkpoint if it exists
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    print(f"Loaded checkpoint from {checkpoint_path}")

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

        # Train the model
        board_tensor = torch.tensor(game.board, dtype=torch.float32).unsqueeze(0)
        policy, value = model(
            board_tensor,
            game.player_captures,
            game.opponent_captures,
            game.num_moves,
        )
        target_value = torch.tensor([game.value], dtype=torch.float32)
        value_loss = torch.nn.functional.mse_loss(value.view(-1), target_value)
        optimizer.zero_grad()
        value_loss.backward()
        optimizer.step()

    if game.value is not None:
        print("Game Over with move:", move, "Number of moves:", game.num_moves)
        break

    # Flip the board for the opponent's turn
    game.board *= -1

# Pretty print the final board state
pretty_print(game.board, move)

# Save the checkpoint
torch.save(
    {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    },
    checkpoint_path,
)

# Print how many records are in the database
total_records = db.get_total(settings.table_name)
print(f"Total records in {settings.table_name}: {total_records}")

rows = db.fetch_collection(settings.table_name, 50)

# for row in rows:
#     id, board, player_captures, opponent_captures, num_moves, policy, value = db.decode_row(row)
#     if value is not None:
#         # print(f"Number of moves: {num_moves}, Value: {value:.2f}")
#         print(f"ID: {id} Num Moves: {num_moves} Value: {value:.2f}")
