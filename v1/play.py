from flask import Flask, render_template, jsonify, request, session
import numpy as np
import random
import secrets
from dataclasses import dataclass
from game import reset_game, get_legal_moves, make_move, check_n_in_a_row
from settings_loader import load_settings
import torch
from models.gomoku_simple_nn import GomokuSimpleNN
from mcts import MCTS
import os

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)


@dataclass
class Settings:
    board_size: list
    captures_enabled: bool
    tournament_rules_enabled: bool
    connect_n: int
    model: str


def load_model(settings):
    try:
        model = GomokuSimpleNN(settings["board_size"][0])
        checkpoint_path = f"./checkpoints/{settings['model']}.pt"

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            print(
                f"Checkpoint loaded successfully: {checkpoint.keys()}"
            )  # Debugging statement
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            return model
        else:
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    except Exception as e:
        print(f"Error in load_model: {e}")  # Debugging statement
        raise


@app.route("/")
def index():
    settings = load_settings()
    board_size = list(settings["board_size"])
    return render_template("index.html", board_size=board_size)


@app.route("/api/new_game", methods=["POST"])
def new_game():
    data = request.json

    # Load game settings from file
    settings = load_settings()

    # Use settings from file, but allow overrides from request
    session["settings"] = {
        "board_size": data.get("board_size", settings["board_size"]),
        "captures_enabled": data.get("captures_enabled", settings["captures_enabled"]),
        "tournament_rules_enabled": data.get(
            "tournament_rules_enabled", settings["tournament_rules_enabled"]
        ),
        "connect_n": data.get("connect_n", settings["connect_n"]),
        "model": settings["model"],
    }

    board, player_captures, opponent_captures = reset_game(settings["board_size"])
    session["board"] = board.tolist()
    session["player_captures"] = player_captures
    session["opponent_captures"] = opponent_captures
    session["num_moves"] = 0

    # Store user color and determine player values
    user_color = data.get("user_color", "white")  # Default to white
    session["user_color"] = user_color
    session["user_value"] = -1 if user_color == "white" else 1
    session["ai_value"] = 1 if user_color == "white" else -1

    # White goes first in this game
    session["current_player"] = -1  # -1 = white, 1 = black

    session["game_over"] = False
    session["winner"] = None
    session["ai_move_type"] = None

    # If AI plays white (user chose black), AI needs to make first move
    needs_ai_move = user_color == "black"

    return jsonify(
        {
            "board": session["board"],
            "player_captures": session["player_captures"],
            "opponent_captures": session["opponent_captures"],
            "current_player": session["current_player"],
            "game_over": False,
            "winner": None,
            "ai_move_type": None,
            "user_color": user_color,
            "user_value": session["user_value"],
            "ai_value": session["ai_value"],
            "needs_ai_move": needs_ai_move,
        }
    )


@app.route("/api/make_move", methods=["POST"])
def make_move_api():
    if "settings" not in session:
        return jsonify({"error": "No game in progress"}), 400

    # Reconstruct game state from session
    settings = session["settings"]
    board = np.array(session["board"])
    player_captures = session["player_captures"]
    opponent_captures = session["opponent_captures"]
    num_moves = session["num_moves"]
    current_player = session["current_player"]
    user_value = session["user_value"]
    ai_value = session["ai_value"]

    # Check if it's actually the user's turn
    if current_player != user_value:
        return jsonify({"error": "It's not your turn"}), 400

    data = request.json
    move = (data["x"], data["y"])

    try:
        # Validate and make the move
        legal_moves = get_legal_moves(
            board, num_moves, settings["tournament_rules_enabled"]
        )
        if move not in legal_moves:
            raise ValueError("Invalid move: Move is not legal.")

        # Make the move with the current player's value
        board[move[0], move[1]] = current_player

        # Check if game is over
        game_over = check_n_in_a_row(move, board, settings["connect_n"], current_player)

        winner = None
        if game_over:
            winner = "user"

        # Check for draw (no legal moves left)
        if not game_over and not get_legal_moves(
            board, num_moves + 1, settings["tournament_rules_enabled"]
        ):
            game_over = True
            winner = "draw"

        # Update session state
        session["board"] = board.tolist()
        session["player_captures"] = player_captures
        session["opponent_captures"] = opponent_captures
        session["num_moves"] = num_moves + 1
        session["current_player"] = -current_player  # Switch turn
        session["game_over"] = game_over
        session["winner"] = winner

        # Determine if AI should move next
        needs_ai_move = not game_over and session["current_player"] == ai_value

        return jsonify(
            {
                "board": session["board"],
                "player_captures": session["player_captures"],
                "opponent_captures": session["opponent_captures"],
                "current_player": session["current_player"],
                "game_over": session["game_over"],
                "winner": session["winner"],
                "ai_move_type": session.get("ai_move_type"),
                "needs_ai_move": needs_ai_move,
            }
        )

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "An unexpected error occurred: " + str(e)}), 500


@app.route("/api/make_ai_move", methods=["POST"])
def make_ai_move():
    if "settings" not in session:
        return jsonify({"error": "No game in progress"}), 400

    # Reconstruct game state from session
    settings = session["settings"]
    board = np.array(session["board"], dtype=np.int8)  # Ensure board is a NumPy array
    player_captures = session["player_captures"]
    opponent_captures = session["opponent_captures"]
    num_moves = session["num_moves"]
    current_player = session["current_player"]
    ai_value = session["ai_value"]

    if current_player != ai_value:  # Ensure it's AI's turn
        return jsonify({"error": "It's not AI's turn"}), 400

    try:
        # Ensure board is a NumPy array
        board = np.array(session["board"], dtype=np.int8)

        # Get legal moves
        legal_moves = get_legal_moves(
            board, num_moves, settings["tournament_rules_enabled"]
        )

        if not legal_moves:
            raise ValueError("No legal moves available")

        # print(board, player_captures, opponent_captures)

        # Use MCTS to determine the best move
        model = load_model(settings)
        mcts = MCTS(model, simulations=100)

        # copy and flip board
        model_board = np.copy(board)
        # model_board *= -1  # Flip the board for the model

        print("board:", model_board)
        move, policy = mcts.best_move(model_board, player_captures, opponent_captures)
        print("best move:", move, "policy:", policy)
        if move is None:
            raise ValueError("No valid move found by MCTS.")

        # Make the move with AI's value
        board[move[0], move[1]] = current_player

        # Check if game is over
        game_over = check_n_in_a_row(move, board, settings["connect_n"])
        winner = None
        if game_over:
            winner = "ai"

        # Check for draw (no legal moves left)
        if not game_over and not get_legal_moves(
            board, num_moves + 1, settings["tournament_rules_enabled"]
        ):
            game_over = True
            winner = "draw"

        # Update session state
        session["board"] = board.tolist()
        session["player_captures"] = player_captures
        session["opponent_captures"] = opponent_captures
        session["num_moves"] = num_moves + 1
        session["current_player"] = -current_player  # Switch turn to user
        session["game_over"] = game_over
        session["winner"] = winner
        session["ai_move_type"] = "mcts"  # Updated to "mcts"

        return jsonify(
            {
                "board": session["board"],
                "player_captures": session["player_captures"],
                "opponent_captures": session["opponent_captures"],
                "current_player": session["current_player"],  # Updated to user
                "game_over": session["game_over"],
                "winner": session["winner"],
                "ai_move_type": session["ai_move_type"],
                "ai_move": move,
                "needs_ai_move": False,  # User's turn now
                "user_value": session["user_value"],  # Ensure user_value is included
            }
        )

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "An unexpected error occurred: " + str(e)}), 500


@app.route("/api/game_types", methods=["GET"])
def get_game_types():
    """Get available game types from settings.json"""
    try:
        import json

        with open("settings.json", "r") as f:
            all_settings = json.load(f)

        game_types = {}
        for key, value in all_settings.items():
            if isinstance(value, dict) and "board_size" in value:
                game_types[key] = {
                    "description": value.get("description", ""),
                    "board_size": value["board_size"],
                    "captures_enabled": value["captures_enabled"],
                    "tournament_rules_enabled": value["tournament_rules_enabled"],
                    "connect_n": value["connect_n"],
                }

        return jsonify(
            {"default": all_settings.get("game", "gomoku_simple"), "games": game_types}
        )
    except Exception as e:
        return jsonify({"error": "Could not read game types"}), 500


# Create the templates directory and HTML file
os.makedirs("templates", exist_ok=True)

html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gomoku Game</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
            background-color: #f0f0f0;
        }
        .game-container {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        h1 {
            text-align: center;
            margin-bottom: 20px;
        }
        .board {
            display: inline-block;
            background-color: #deb887;
            padding: 10px;
            border-radius: 5px;
            margin: 0 auto;
        }
        .board-row {
            display: flex;
            justify-content: center;
        }
        .cell {
            width: 30px;
            height: 30px;
            background-color: #f5deb3;
            border: 1px solid #8b7355;
            cursor: pointer;
            position: relative;
            margin: 0.5px;
        }
        .cell:hover {
            background-color: #ffd700;
        }
        .cell.disabled {
            cursor: default; /* Updated to show default cursor instead of "not-allowed" */
        }
        .cell.disabled:hover {
            background-color: #f5deb3;
        }
        .stone {
            width: 26px;
            height: 26px;
            border-radius: 50%;
            position: absolute;
            top: 2px;
            left: 2px;
        }
        .black-stone {
            background-color: #000;
            box-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .white-stone {
            background-color: #fff;
            border: 1px solid #ccc;
            box-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .controls {
            margin-top: 20px;
            text-align: center;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            margin: 0 5px;
        }
        button:hover {
            background-color: #45a049;
        }
        button:disabled {
            background-color: #cccccc;
            cursor: not-allowed;
        }
        .info {
            margin-top: 20px;
            text-align: center;
            width: 100%;
        }
        .captures {
            display: flex;
            justify-content: space-around;
            margin-top: 10px;
        }
        .status {
            font-size: 18px;
            font-weight: bold;
            margin-top: 10px;
        }
        .winner {
            color: #4CAF50;
        }
        .settings {
            margin-bottom: 20px;
            text-align: center;
            width: 100%;
        }
        .settings label {
            margin: 0 10px;
        }
        .game-selector {
            margin-bottom: 10px;
        }
        .color-indicator {
            display: inline-block;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            vertical-align: middle;
            margin-left: 5px;
        }
        .color-indicator.black {
            background-color: #000;
            border: 1px solid #333;
        }
        .color-indicator.white {
            background-color: #fff;
            border: 1px solid #ccc;
        }
        .thinking {
            animation: pulse 1s infinite;
        }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
    </style>
</head>
<body>
    <div class="game-container">
        <h1>Gomoku - Play vs AI</h1>
        
        <div class="settings">
            <div class="game-selector">
                <label>
                    Game Type: 
                    <select id="gameType" onchange="updateGameSettings()">
                        <option value="">Default (from settings.json)</option>
                        <option value="gomoku">Gomoku (19x19)</option>
                        <option value="gomoku_simple">Gomoku Simple (7x7)</option>
                        <option value="pente">Pente (19x19)</option>
                    </select>
                </label>
            </div>
            <label>
                <input type="checkbox" id="capturesEnabled" checked> Enable Captures
            </label>
            <label>
                <input type="checkbox" id="tournamentRules" checked> Tournament Rules
            </label>
            <label>
                Connect: <input type="number" id="connectN" value="5" min="3" max="8" style="width: 40px;">
            </label>
            <label>
                Your Color: 
                <select id="userColor">
                    <option value="white">White (goes first)</option>
                    <option value="black">Black (goes second)</option>
                </select>
            </label>
        </div>
        
        <div class="board" id="board"></div>
        
        <div class="controls">
            <button onclick="newGame()">New Game</button>
        </div>
        
        <div class="info">
            <div class="captures">
                <div>Your Captures: <span id="playerCaptures">0</span></div>
                <div>AI Captures: <span id="aiCaptures">0</span></div>
            </div>
            <div class="status" id="status">Click New Game to start!</div>
        </div>
    </div>

    <script>
        let gameState = null;
        let boardSize = {{ board_size }};
        let isProcessing = false;

        function updateGameSettings() {
            const gameType = document.getElementById('gameType').value;
            if (gameType === 'pente') {
                document.getElementById('capturesEnabled').checked = true;
            } else if (gameType === 'gomoku' || gameType === 'gomoku_simple') {
                document.getElementById('capturesEnabled').checked = false;
            }
        }

        function createBoard() {
            const board = document.getElementById('board');
            board.innerHTML = '';
            
            for (let i = 0; i < boardSize[0]; i++) {
                const row = document.createElement('div');
                row.className = 'board-row';
                
                for (let j = 0; j < boardSize[1]; j++) {
                    const cell = document.createElement('div');
                    cell.className = 'cell';
                    cell.dataset.x = i;
                    cell.dataset.y = j;
                    cell.onclick = () => makeMove(i, j);
                    row.appendChild(cell);
                }
                board.appendChild(row);
            }
        }

        function updateBoard() {
            if (!gameState) return;
            
            const cells = document.querySelectorAll('.cell');
            cells.forEach(cell => {
                const x = parseInt(cell.dataset.x);
                const y = parseInt(cell.dataset.y);
                const value = gameState.board[x][y];
                
                cell.innerHTML = '';
                if (value !== 0) {
                    const stone = document.createElement('div');
                    stone.className = 'stone ' + (value === 1 ? 'black-stone' : 'white-stone');
                    cell.appendChild(stone);
                }
                
                // Disable clicks during AI turn or when game is over
                if (isProcessing || gameState.game_over || 
                    (gameState.current_player !== gameState.user_value)) {
                    cell.classList.add('disabled');
                } else {
                    cell.classList.remove('disabled');
                }
            });
            
            document.getElementById('playerCaptures').textContent = gameState.player_captures;
            document.getElementById('aiCaptures').textContent = gameState.opponent_captures;
            
            updateStatus();
        }

        function updateStatus() {
            const statusEl = document.getElementById('status');
            
            if (gameState.game_over) {
                if (gameState.winner === 'user') {
                    statusEl.innerHTML = '<span class="winner">You Win! 🎉</span>';
                } else if (gameState.winner === 'ai') {
                    statusEl.innerHTML = 'AI Wins! 🤖';
                } else if (gameState.winner === 'draw') {
                    statusEl.innerHTML = 'Draw! 🤝';
                }
            } else {
                const isUserTurn = gameState.current_player === gameState.user_value;
                const userColor = gameState.user_color;
                const currentColor = gameState.current_player === 1 ? 'black' : 'white';
                
                if (isUserTurn) {
                    statusEl.innerHTML = `Your turn <span class="color-indicator ${userColor}"></span>`;
                } else {
                    statusEl.innerHTML = `<span class="thinking">AI thinking...</span> <span class="color-indicator ${currentColor}"></span>`;
                }
            }
        }

        async function newGame() {
            isProcessing = true;
            
            const settings = {
                game_type: document.getElementById('gameType').value || null,
                captures_enabled: document.getElementById('capturesEnabled').checked,
                tournament_rules_enabled: document.getElementById('tournamentRules').checked,
                connect_n: parseInt(document.getElementById('connectN').value),
                user_color: document.getElementById('userColor').value
            };
            
            try {
                const response = await fetch('/api/new_game', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(settings)
                });
                
                if (response.ok) {
                    gameState = await response.json();
                    
                    // Update board size if it changed
                    if (gameState.board.length !== boardSize[0] || 
                        (gameState.board[0] && gameState.board[0].length !== boardSize[1])) {
                        boardSize = [gameState.board.length, gameState.board[0].length];
                        createBoard();
                    }
                    
                    updateBoard();
                    
                    // If AI needs to move first (user chose white)
                    if (gameState.needs_ai_move) {
                        await makeAIMove();
                    }
                } else {
                    const error = await response.json();
                    console.error('Error starting game:', error.error);
                }
            } finally {
                isProcessing = false;
            }
        }

        async function makeMove(x, y) {
            if (!gameState || gameState.game_over || isProcessing) {
                return;
            }
            
            // Check if it's the user's turn
            if (gameState.current_player !== gameState.user_value) {
                console.log("Not your turn!", gameState.current_player, gameState.user_value);
                return;
            }
            
            isProcessing = true;
            
            try {
                const response = await fetch('/api/make_move', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({x, y})
                });
                
                if (response.ok) {
                    gameState = await response.json();
                    updateBoard();
                    
                    // If game is not over and it's AI's turn, make AI move
                    if (!gameState.game_over && gameState.needs_ai_move) {
                        await makeAIMove();
                    }
                } else {
                    const error = await response.json();
                    console.error('Error making move:', error.error);
                }
            } finally {
                isProcessing = false;
            }
        }

        async function makeAIMove() {
            if (!gameState || gameState.game_over) return;
            
            updateStatus(); // Show "AI thinking..."
            
            // Add a small delay to make it feel more natural
            await new Promise(resolve => setTimeout(resolve, 500));
            
            try {
                const response = await fetch('/api/make_ai_move', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({})
                });
                
                if (response.ok) {
                    gameState = await response.json();
                    updateBoard();
                } else {
                    const error = await response.json();
                    console.error('Error with AI move:', error.error);
                }
            } catch (error) {
                console.error('Error making AI move:', error);
            }
        }

        // Initialize
        createBoard();
    </script>
</body>
</html>"""

with open("templates/index.html", "w") as f:
    f.write(html_content)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
