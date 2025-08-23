import threading
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from games.pente import Pente
from players import HumanPlayer, EasyMCTSPlayer, MediumMCTSPlayer, HardMCTSPlayer, ExpertMCTSPlayer

@dataclass
class GameState:
    """Serializable game state for web interface."""
    board: List[List[int]]
    current_player: int
    captures: Dict[int, int]
    game_over: bool
    winner: Optional[int]
    legal_moves: List[Tuple[int, int]]
    move_history: List[Tuple[int, int]]
    human_player_id: int
    ai_player_id: int
    waiting_for_human: bool
    ai_thinking: bool
    tournament_rule: bool
    captures_to_win: int

class GameSession:
    """Manages a single game session."""
    
    def __init__(self, session_id: str, board_size: int = 19, captures_to_win: int = 5,
                 tournament_rule: bool = True, human_player_id: int = 1, ai_difficulty: str = 'medium'):
        self.session_id = session_id
        self.game = Pente(board_size=board_size, captures_to_win=captures_to_win, tournament_rule=tournament_rule)
        self.human_player_id = human_player_id
        self.ai_player_id = -human_player_id
        
        # Create players
        self.human_player = HumanPlayer(name="You", player_id=human_player_id)
        self.ai_player = self._create_ai_player(ai_difficulty, self.ai_player_id)
        
        # Game state
        self.ai_thinking = False
        self.last_activity = time.time()
        self._lock = threading.Lock()
    
    def _create_ai_player(self, difficulty: str, player_id: int):
        """Create AI player based on difficulty setting."""
        difficulty_map = {
            'easy': EasyMCTSPlayer,
            'medium': MediumMCTSPlayer,
            'hard': HardMCTSPlayer,
            'expert': ExpertMCTSPlayer
        }
        
        player_class = difficulty_map.get(difficulty.lower(), MediumMCTSPlayer)
        return player_class(player_id=player_id)
    
    def get_current_player(self):
        """Get the player object for current turn."""
        if self.game.current_player == self.human_player_id:
            return self.human_player
        else:
            return self.ai_player
    
    def is_human_turn(self) -> bool:
        """Check if it's the human player's turn."""
        return self.game.current_player == self.human_player_id
    
    def make_move(self, move: Tuple[int, int]) -> None:
        """Make a move in the game."""
        with self._lock:
            self.game.make_move(move)
            self.last_activity = time.time()
            
            # Notify players
            self.human_player.notify_move(move, self.game.current_player)
            self.ai_player.notify_move(move, self.game.current_player)
    
    def get_ai_move(self) -> Tuple[int, int]:
        """Get AI move using current AI player."""
        with self._lock:
            self.ai_thinking = True
        
        try:
            move = self.ai_player.get_move(self.game)
            return move
        finally:
            with self._lock:
                self.ai_thinking = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert game session to dictionary for JSON serialization."""
        with self._lock:
            # Convert numpy board to regular list
            board = self.game.board.tolist()
            
            return asdict(GameState(
                board=board,
                current_player=self.game.current_player,
                captures=self.game.captures,
                game_over=self.game.is_terminal(),
                winner=self.game.get_winner(),
                legal_moves=self.game.get_legal_moves(),
                move_history=self.game.move_history.copy(),
                human_player_id=self.human_player_id,
                ai_player_id=self.ai_player_id,
                waiting_for_human=self.is_human_turn() and not self.game.is_terminal(),
                ai_thinking=self.ai_thinking,
                tournament_rule=self.game.tournament_rule,
                captures_to_win=self.game.captures_to_win
            ))

class GameController:
    """Manages multiple game sessions."""
    
    def __init__(self):
        self.sessions: Dict[str, GameSession] = {}
        self._lock = threading.Lock()
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_sessions, daemon=True)
        self._cleanup_thread.start()
    
    def new_game(self, session_id: str, board_size: int = 19, captures_to_win: int = 5,
                 tournament_rule: bool = True, human_player_id: int = 1, ai_difficulty: str = 'medium') -> Dict[str, Any]:
        """Create a new game session."""
        with self._lock:
            # Clean up old session if exists
            if session_id in self.sessions:
                old_session = self.sessions[session_id]
                if hasattr(old_session.human_player, 'cancel_move_request'):
                    old_session.human_player.cancel_move_request()
            
            # Create new session
            session = GameSession(
                session_id=session_id,
                board_size=board_size,
                captures_to_win=captures_to_win,
                tournament_rule=tournament_rule,
                human_player_id=human_player_id,
                ai_difficulty=ai_difficulty
            )
            
            # Automatically place the first move at center (player 1 always goes first)
            center = board_size // 2
            session.make_move((center, center))
            
            self.sessions[session_id] = session
            return session.to_dict()
    
    def get_game_state(self, session_id: str) -> Dict[str, Any]:
        """Get current game state for session."""
        with self._lock:
            if session_id not in self.sessions:
                raise ValueError("Session not found")
            
            return self.sessions[session_id].to_dict()
    
    def make_human_move(self, session_id: str, move: Tuple[int, int]) -> Dict[str, Any]:
        """Process human player move."""
        with self._lock:
            if session_id not in self.sessions:
                raise ValueError("Session not found")
            
            session = self.sessions[session_id]
            
            # Validate it's human's turn
            if not session.is_human_turn():
                raise ValueError("Not human player's turn")
            
            if session.game.is_terminal():
                raise ValueError("Game is already over")
            
            # Validate move is legal with specific error messages
            legal_moves = session.game.get_legal_moves()
            if move not in legal_moves:
                row, col = move
                
                # Check if position is already occupied
                if session.game.board[row, col] != 0:
                    raise ValueError("Position already occupied")
                
                # Check for tournament rule violation
                if (session.game.tournament_rule and 
                    session.game.move_count == 2 and 
                    session.game.current_player == 1):
                    center = session.game.board_size // 2
                    distance = max(abs(row - center), abs(col - center))
                    if distance < 3:
                        raise ValueError("Tournament rule: Second move must be at least 3 intersections from center")
                
                # Check if position is out of bounds
                if row < 0 or row >= session.game.board_size or col < 0 or col >= session.game.board_size:
                    raise ValueError("Move out of bounds")
                
                # Generic invalid move
                raise ValueError("Invalid move")
            
            # Make the move
            session.make_move(move)
            
            return session.to_dict()
    
    def make_ai_move(self, session_id: str) -> Dict[str, Any]:
        """Process AI player move."""
        if session_id not in self.sessions:
            raise ValueError("Session not found")
        
        session = self.sessions[session_id]
        
        # Validate it's AI's turn
        if session.is_human_turn():
            raise ValueError("Not AI player's turn")
        
        if session.game.is_terminal():
            raise ValueError("Game is already over")
        
        # Get AI move (this may take time)
        def make_ai_move_async():
            try:
                move = session.get_ai_move()
                session.make_move(move)
            except Exception as e:
                print(f"Error in AI move: {e}")
        
        # Run AI move in background for web interface
        ai_thread = threading.Thread(target=make_ai_move_async)
        ai_thread.start()
        ai_thread.join()  # Wait for completion
        
        with self._lock:
            return session.to_dict()
    
    def get_move_statistics(self, session_id: str) -> List[Tuple[Tuple[int, int], int, float]]:
        """Get AI move analysis for current position."""
        with self._lock:
            if session_id not in self.sessions:
                raise ValueError("Session not found")
            
            session = self.sessions[session_id]
            
            # Only get stats if AI player supports it
            if hasattr(session.ai_player, 'get_move_statistics'):
                return session.ai_player.get_move_statistics(session.game)
            else:
                return []
    
    def _cleanup_sessions(self):
        """Clean up old inactive sessions."""
        while True:
            try:
                time.sleep(300)  # Check every 5 minutes
                current_time = time.time()
                
                with self._lock:
                    inactive_sessions = []
                    for session_id, session in self.sessions.items():
                        # Remove sessions inactive for more than 1 hour
                        if current_time - session.last_activity > 3600:
                            inactive_sessions.append(session_id)
                    
                    for session_id in inactive_sessions:
                        session = self.sessions[session_id]
                        if hasattr(session.human_player, 'cancel_move_request'):
                            session.human_player.cancel_move_request()
                        del self.sessions[session_id]
                        
            except Exception as e:
                print(f"Error in session cleanup: {e}")