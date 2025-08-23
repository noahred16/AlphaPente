from flask import Flask, render_template, jsonify, request, session
import secrets
import threading
from typing import Dict, Optional
from .game_controller import GameController

def create_app() -> Flask:
    """Create and configure Flask application."""
    app = Flask(__name__, 
                template_folder='templates',
                static_folder='static')
    app.secret_key = secrets.token_hex(16)
    
    # Global game controller to manage sessions
    game_controller = GameController()
    
    @app.route('/')
    def index():
        """Main game page."""
        return render_template('index.html')
    
    @app.route('/api/new_game', methods=['POST'])
    def new_game():
        """Start a new game with specified settings."""
        try:
            data = request.get_json() or {}
            
            # Get session ID
            if 'session_id' not in session:
                session['session_id'] = secrets.token_hex(16)
            
            session_id = session['session_id']
            
            # Game settings
            board_size = data.get('board_size', 19)
            captures_to_win = data.get('captures_to_win', 5)
            tournament_rule = data.get('tournament_rule', True)
            human_player_id = 1 if data.get('human_goes_first', True) else -1
            ai_difficulty = data.get('ai_difficulty', 'medium')
            
            # Create new game
            game_state = game_controller.new_game(
                session_id=session_id,
                board_size=board_size,
                captures_to_win=captures_to_win,
                tournament_rule=tournament_rule,
                human_player_id=human_player_id,
                ai_difficulty=ai_difficulty
            )
            
            return jsonify(game_state)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    
    @app.route('/api/make_move', methods=['POST'])
    def make_move():
        """Submit a human player move."""
        try:
            data = request.get_json()
            session_id = session.get('session_id')
            
            if not session_id:
                return jsonify({'error': 'No active session'}), 400
            
            row = data.get('row')
            col = data.get('col')
            
            if row is None or col is None:
                return jsonify({'error': 'Missing move coordinates'}), 400
            
            # Submit move and get updated game state
            game_state = game_controller.make_human_move(
                session_id=session_id,
                move=(row, col)
            )
            
            return jsonify(game_state)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    
    @app.route('/api/game_state', methods=['GET'])
    def get_game_state():
        """Get current game state."""
        try:
            session_id = session.get('session_id')
            
            if not session_id:
                return jsonify({'error': 'No active session'}), 400
            
            game_state = game_controller.get_game_state(session_id)
            return jsonify(game_state)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    
    @app.route('/api/ai_move', methods=['POST'])
    def trigger_ai_move():
        """Trigger AI move (if it's AI's turn)."""
        try:
            session_id = session.get('session_id')
            
            if not session_id:
                return jsonify({'error': 'No active session'}), 400
            
            game_state = game_controller.make_ai_move(session_id)
            return jsonify(game_state)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    
    @app.route('/api/move_statistics', methods=['GET'])
    def get_move_statistics():
        """Get AI move analysis for current position."""
        try:
            session_id = session.get('session_id')
            
            if not session_id:
                return jsonify({'error': 'No active session'}), 400
            
            stats = game_controller.get_move_statistics(session_id)
            return jsonify({'statistics': stats})
            
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    
    return app

# For development
if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='127.0.0.1', port=5000)