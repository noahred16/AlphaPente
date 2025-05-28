from flask import Flask, render_template, jsonify, request, session
import numpy as np
import random
import secrets
from dataclasses import dataclass
from game import Game
from settings_loader import GameSettings

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

@dataclass
class Settings:
    board_size: list
    captures_enabled: bool
    tournament_rules_enabled: bool
    connect_n: int

# Helper function to flip the board for opponent's perspective
def flip_board(board):
    """Flip the board values (1 -> -1, -1 -> 1, 0 -> 0)"""
    return -board

# Helper function to convert move coordinates when board is flipped
def flip_move(move, board_size):
    """Flip move coordinates - in this case, just return the same coordinates"""
    # Since we're just flipping values, not rotating the board, coordinates stay the same
    return move

@app.route('/')
def index():
    # Load settings to get board size
    game_settings = GameSettings()
    board_size = list(game_settings.board_size)
    return render_template('index.html', board_size=board_size)

@app.route('/api/new_game', methods=['POST'])
def new_game():
    data = request.json
    
    # Load game settings from file
    game_type = data.get('game_type', None)  # Allow specifying game type
    game_settings = GameSettings(game_type)
    
    # Use settings from file, but allow overrides from request
    settings = Settings(
        board_size=list(game_settings.board_size),
        captures_enabled=data.get('captures_enabled', game_settings.captures_enabled),
        tournament_rules_enabled=data.get('tournament_rules_enabled', game_settings.tournament_rules_enabled),
        connect_n=data.get('connect_n', game_settings.connect_n)
    )
    
    game = Game(settings)
    
    # Store game settings and state in session
    session['settings'] = {
        'board_size': settings.board_size,
        'captures_enabled': settings.captures_enabled,
        'tournament_rules_enabled': settings.tournament_rules_enabled,
        'connect_n': settings.connect_n
    }
    session['board'] = game.board.tolist()
    session['player_captures'] = game.player_captures
    session['opponent_captures'] = game.opponent_captures
    session['num_moves'] = game.num_moves
    session['current_player'] = 1  # 1 for human, -1 for AI
    session['game_over'] = False
    session['winner'] = None
    
    return jsonify({
        'board': game.board.tolist(),
        'player_captures': game.player_captures,
        'opponent_captures': game.opponent_captures,
        'current_player': session['current_player'],
        'game_over': False,
        'winner': None
    })

@app.route('/api/make_move', methods=['POST'])
def make_move():
    if 'settings' not in session:
        return jsonify({'error': 'No game in progress'}), 400
    
    # Reconstruct game from session
    settings = Settings(**session['settings'])
    game = Game(settings)
    game.board = np.array(session['board'])
    game.player_captures = session['player_captures']
    game.opponent_captures = session['opponent_captures']
    game.num_moves = session['num_moves']
    
    current_player = session['current_player']
    
    data = request.json
    move = (data['x'], data['y'])
    
    try:
        # For AI moves, we need to flip the board
        if current_player == -1:
            game.board = flip_board(game.board)
        
        # Validate and make the move
        legal_moves = game.get_legal_moves()
        if move not in legal_moves:
            raise ValueError("Invalid move: Move is not legal.")
        
        game.make_move(move)
        
        # Check if game is over
        game_over = game.value is not None
        winner = None
        if game_over:
            winner = current_player
        
        # Flip board back if it was AI's turn
        if current_player == -1:
            game.board = flip_board(game.board)
        
        # Update session state
        session['board'] = game.board.tolist()
        session['player_captures'] = game.player_captures
        session['opponent_captures'] = game.opponent_captures
        session['num_moves'] = game.num_moves
        session['game_over'] = game_over
        session['winner'] = winner
        
        # If game is not over and it was human's turn, make AI move
        if not game_over and current_player == 1:
            session['current_player'] = -1
            
            # Flip board for AI's perspective
            game.board = flip_board(game.board)
            
            # Get legal moves and make random AI move
            ai_legal_moves = game.get_legal_moves()
            if ai_legal_moves:
                ai_move = random.choice(ai_legal_moves)
                game.make_move(ai_move)
                
                # Check if AI won
                if game.value is not None:
                    game_over = True
                    winner = -1
                    session['game_over'] = game_over
                    session['winner'] = winner
                
                # Flip board back to human's perspective
                game.board = flip_board(game.board)
                
                # Update session state after AI move
                session['board'] = game.board.tolist()
                session['player_captures'] = game.player_captures
                session['opponent_captures'] = game.opponent_captures
                session['num_moves'] = game.num_moves
            
            session['current_player'] = 1
        else:
            # Switch player if game continues
            session['current_player'] = -current_player
        
        return jsonify({
            'board': session['board'],
            'player_captures': session['player_captures'],
            'opponent_captures': session['opponent_captures'],
            'current_player': session['current_player'],
            'game_over': session['game_over'],
            'winner': session['winner']
        })
        
    except ValueError as e:
        # Restore original board state if move failed
        if current_player == -1:
            game.board = flip_board(game.board)
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': 'An unexpected error occurred'}), 500

@app.route('/api/game_state', methods=['GET'])
def game_state():
    if 'settings' not in session:
        return jsonify({'error': 'No game in progress'}), 400
    
    return jsonify({
        'board': session['board'],
        'player_captures': session['player_captures'],
        'opponent_captures': session['opponent_captures'],
        'current_player': session['current_player'],
        'game_over': session.get('game_over', False),
        'winner': session.get('winner', None)
    })

@app.route('/api/game_types', methods=['GET'])
def get_game_types():
    """Get available game types from settings.json"""
    try:
        import json
        with open("settings.json", "r") as f:
            all_settings = json.load(f)
        
        game_types = {}
        for key, value in all_settings.items():
            if isinstance(value, dict) and 'board_size' in value:
                game_types[key] = {
                    'description': value.get('description', ''),
                    'board_size': value['board_size'],
                    'captures_enabled': value['captures_enabled'],
                    'tournament_rules_enabled': value['tournament_rules_enabled'],
                    'connect_n': value['connect_n']
                }
        
        return jsonify({
            'default': all_settings.get('game', 'gomoku-simple'),
            'games': game_types
        })
    except Exception as e:
        return jsonify({'error': 'Could not read game types'}), 500

# Create the templates directory and HTML file
import os
os.makedirs('templates', exist_ok=True)

html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pente Game</title>
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
    </style>
</head>
<body>
    <div class="game-container">
        <h1>Board Game - Play vs Random AI</h1>
        
        <div class="settings">
            <div class="game-selector">
                <label>
                    Game Type: 
                    <select id="gameType" onchange="updateGameSettings()">
                        <option value="">Default (from settings.json)</option>
                        <option value="gomoku">Gomoku (19x19)</option>
                        <option value="gomoku-simple">Gomoku Simple (7x7)</option>
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
        const boardSize = {{ board_size }};  // This will be injected from Flask

        function updateGameSettings() {
            // This could be extended to dynamically update settings based on game type
            const gameType = document.getElementById('gameType').value;
            if (gameType === 'pente') {
                document.getElementById('capturesEnabled').checked = true;
            } else if (gameType === 'gomoku' || gameType === 'gomoku-simple') {
                document.getElementById('capturesEnabled').checked = false;
            }
        }

        function createBoard() {
            const board = document.getElementById('board');
            board.innerHTML = '';
            
            // Create board dynamically based on boardSize
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
            });
            
            document.getElementById('playerCaptures').textContent = gameState.player_captures;
            document.getElementById('aiCaptures').textContent = gameState.opponent_captures;
            
            updateStatus();
        }

        function updateStatus() {
            const statusEl = document.getElementById('status');
            
            if (gameState.game_over) {
                if (gameState.winner === 1) {
                    statusEl.innerHTML = '<span class="winner">You Win!</span>';
                } else if (gameState.winner === -1) {
                    statusEl.innerHTML = 'AI Wins!';
                } else {
                    statusEl.innerHTML = 'Draw!';
                }
            } else {
                statusEl.textContent = gameState.current_player === 1 ? 'Your turn' : 'AI thinking...';
            }
        }

        async function newGame() {
            const settings = {
                game_type: document.getElementById('gameType').value || null,
                captures_enabled: document.getElementById('capturesEnabled').checked,
                tournament_rules_enabled: document.getElementById('tournamentRules').checked,
                connect_n: parseInt(document.getElementById('connectN').value)
            };
            
            const response = await fetch('/api/new_game', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(settings)
            });
            
            if (response.ok) {
                gameState = await response.json();
                
                // If board size changed, recreate the board
                if (gameState.board.length !== boardSize[0] || 
                    (gameState.board[0] && gameState.board[0].length !== boardSize[1])) {
                    boardSize[0] = gameState.board.length;
                    boardSize[1] = gameState.board[0].length;
                    createBoard();
                }
                
                updateBoard();
            } else {
                const error = await response.json();
                // alert('Error starting game: ' + error.error);
                console.error('Error starting game:', error.error);
            }
        }

        async function makeMove(x, y) {
            console.log(`Making move at (${x}, ${y})`);
            if (!gameState || gameState.game_over || gameState.current_player !== 1) {
                console.warn("Cannot make move: Game is over or it's not your turn.", gameState);
                return;
            }
            
            const response = await fetch('/api/make_move', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({x, y})
            });
            
            if (response.ok) {
                gameState = await response.json();
                updateBoard();
            } else {
                const error = await response.json();
                // alert(error.error);
                console.error('Error making move:', error.error);
            }
        }

        // Initialize
        createBoard();
        newGame();
    </script>
</body>
</html>'''

with open('templates/index.html', 'w') as f:
    f.write(html_content)

if __name__ == '__main__':
    app.run(debug=True, port=5000)