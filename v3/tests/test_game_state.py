"""
Comprehensive tests for GameState class.
"""

import pytest
from src.core.game_state import GameState
from src.core.move_delta import MoveDelta

class TestGameState:
    """Test suite for GameState functionality."""
    
    def test_initialization(self):
        """Test GameState initialization."""
        game = GameState()
        
        # Test default values
        assert game.board_size == 19
        assert game.captures_to_win == 5
        assert game.tournament_rule == True
        assert game.current_player == 1
        assert game.move_count == 0
        assert len(game.move_history) == 0
        assert game.captures[1] == 0
        assert game.captures[-1] == 0
    
    def test_initialization_custom_params(self):
        """Test GameState with custom parameters."""
        game = GameState(board_size=15, captures_to_win=3, tournament_rule=False)
        
        assert game.board_size == 15
        assert game.captures_to_win == 3
        assert game.tournament_rule == False
    
    def test_basic_move_and_undo(self):
        """Test basic move making and undoing."""
        game = GameState()
        
        # Make a move
        delta = game.make_move((9, 9))
        
        assert isinstance(delta, MoveDelta)
        assert delta.position == (9, 9)
        assert delta.player == 1
        assert not delta.has_captures()
        
        assert game.board.get_stone(9, 9) == 1
        assert game.current_player == -1
        assert game.move_count == 1
        assert len(game.move_history) == 1
        
        # Undo the move
        undone_delta = game.undo_move()
        
        assert undone_delta == delta
        assert game.board.get_stone(9, 9) == 0
        assert game.current_player == 1
        assert game.move_count == 0
        assert len(game.move_history) == 0
    
    def test_multiple_moves_and_undos(self):
        """Test sequence of moves and undos."""
        game = GameState()
        
        moves = [(9, 9), (10, 10), (8, 8), (11, 11)]
        deltas = []
        
        # Make moves
        for move in moves:
            delta = game.make_move(move)
            deltas.append(delta)
        
        assert game.move_count == 4
        assert game.current_player == 1  # Started with 1, so after 4 moves back to 1
        
        # Verify stones are placed
        for i, move in enumerate(moves):
            expected_player = 1 if i % 2 == 0 else -1
            assert game.board.get_stone(*move) == expected_player
        
        # Undo all moves
        for i in range(len(moves)):
            expected_delta = deltas[-(i+1)]  # Reverse order
            undone_delta = game.undo_move()
            assert undone_delta == expected_delta
        
        # Game should be back to initial state
        assert game.move_count == 0
        assert game.current_player == 1
        assert len(game.move_history) == 0
        for move in moves:
            assert game.board.get_stone(*move) == 0
    
    def test_invalid_move(self):
        """Test making invalid moves."""
        game = GameState()
        
        # Place a stone
        game.make_move((5, 5))
        
        # Try to place on same position
        with pytest.raises(ValueError, match="already occupied"):
            game.make_move((5, 5))
    
    def test_undo_empty_history(self):
        """Test undoing when no moves have been made."""
        game = GameState()
        
        result = game.undo_move()
        assert result is None
    
    def test_capture_detection(self):
        """Test capture detection and handling."""
        game = GameState()
        
        # Set up a capture scenario: X-O-O-X pattern  
        # We need to create: X at (5,5), O at (5,6), O at (5,7), then X at (5,8) to capture
        game.make_move((5, 5))   # Player 1 (X)
        game.make_move((10, 10)) # Player -1 (O) - unrelated move
        game.make_move((5, 8))   # Player 1 (X) - the capturing stone position
        game.make_move((5, 6))   # Player -1 (O) - first stone to be captured
        game.make_move((9, 9))   # Player 1 (X) - unrelated move
        game.make_move((5, 7))   # Player -1 (O) - second stone to be captured
        
        # Now we have: X at (5,5), O at (5,6), O at (5,7), X at (5,8)
        # This should already be a capture. Let me fix this by setting up the board directly
        game = GameState()
        game.board.set_stone(5, 5, 1)  # Player 1
        game.board.set_stone(5, 6, -1) # Player -1  
        game.board.set_stone(5, 7, -1) # Player -1
        
        # Now player 1 plays at (5, 8) to create X-O-O-X capture
        game.current_player = 1
        delta = game.make_move((5, 8))
        
        assert delta.has_captures()
        assert len(delta.captured_positions) == 2  # Two stones captured
        assert delta.capture_count == 1  # One pair
        assert game.captures[1] == 1  # Player 1 has 1 capture
        
        # Verify captured stones are removed
        assert game.board.get_stone(5, 6) == 0
        assert game.board.get_stone(5, 7) == 0
        
        # Undo and verify restoration
        game.undo_move()
        assert game.board.get_stone(5, 6) == -1
        assert game.board.get_stone(5, 7) == -1
        assert game.captures[1] == 0
    
    def test_legal_moves_basic(self):
        """Test basic legal move generation."""
        game = GameState(board_size=5, tournament_rule=False)  # Disable tournament rule for basic test
        
        # Initially all positions should be legal
        legal_moves = game.get_legal_moves()
        assert len(legal_moves) == 25  # 5x5 board
        
        # Make a move
        game.make_move((2, 2))
        legal_moves = game.get_legal_moves()
        assert len(legal_moves) == 24  # One position occupied
        assert (2, 2) not in legal_moves
    
    def test_tournament_rule(self):
        """Test tournament rule enforcement."""
        game = GameState(board_size=19, tournament_rule=True)
        
        # First move (player 1) should allow all positions
        legal_moves = game.get_legal_moves()
        assert len(legal_moves) == 361  # All positions
        
        # Make first move at center
        game.make_move((9, 9))
        
        # Second move (player -1) should be restricted to ≥3 from center
        legal_moves = game.get_legal_moves()
        center = 9
        
        # Verify all legal moves are ≥3 from center
        for row, col in legal_moves:
            distance = max(abs(row - center), abs(col - center))
            assert distance >= 3
        
        # Positions close to center should not be legal
        close_positions = [(9, 10), (10, 9), (8, 9), (9, 8), (10, 10)]
        for pos in close_positions:
            assert pos not in legal_moves
    
    def test_tournament_rule_disabled(self):
        """Test game without tournament rule."""
        game = GameState(tournament_rule=False)
        
        # Make first move
        game.make_move((9, 9))
        
        # All empty positions should be legal (no restriction)
        legal_moves = game.get_legal_moves()
        assert len(legal_moves) == 360  # 361 - 1 occupied
        assert (9, 10) in legal_moves  # Adjacent to center should be legal
    
    def test_five_in_row_horizontal(self):
        """Test horizontal five-in-a-row detection."""
        game = GameState()
        
        # Place 5 stones horizontally
        for i in range(5):
            game.make_move((10, 5 + i))
            if i < 4:  # Don't place opponent stone after last move
                game.make_move((11, 5 + i))  # Opponent moves
        
        assert game.get_winner() == 1
        assert game.is_terminal()
    
    def test_five_in_row_vertical(self):
        """Test vertical five-in-a-row detection."""
        game = GameState()
        
        # Place 5 stones vertically
        for i in range(5):
            game.make_move((5 + i, 10))
            if i < 4:
                game.make_move((5 + i, 11))  # Opponent moves
        
        assert game.get_winner() == 1
        assert game.is_terminal()
    
    def test_five_in_row_diagonal(self):
        """Test diagonal five-in-a-row detection."""
        game = GameState()
        
        # Place 5 stones diagonally
        for i in range(5):
            game.make_move((5 + i, 5 + i))
            if i < 4:
                game.make_move((5 + i, 6 + i))  # Opponent moves
        
        assert game.get_winner() == 1
        assert game.is_terminal()
    
    def test_capture_win(self):
        """Test winning by captures."""
        game = GameState(captures_to_win=2)  # Lower threshold for testing
        
        # Simulate capturing 2 pairs (4 stones total)
        # This is complex to set up naturally, so we'll modify captures directly for testing
        game.captures[1] = 2
        
        assert game.get_winner() == 1
        assert game.is_terminal()
    
    def test_no_winner_ongoing_game(self):
        """Test game with no winner yet."""
        game = GameState()
        
        # Place some stones but no five-in-a-row or captures
        game.make_move((5, 5))
        game.make_move((10, 10))
        game.make_move((15, 15))
        
        assert game.get_winner() is None
        assert not game.is_terminal()
    
    def test_count_line_method(self):
        """Test the _count_line helper method."""
        game = GameState()
        
        # Place stones in a line for player 1
        player1_positions = [(10, 5), (10, 6), (10, 7)]
        for pos in player1_positions:
            game.board.set_stone(*pos, 1)
            
        # Place stones in a line for player -1
        player2_positions = [(12, 5), (12, 6)]
        for pos in player2_positions:
            game.board.set_stone(*pos, -1)
        
        # Test counting player 1 line from middle
        count = game._count_line(10, 6, 0, 1, 1)  # Player 1, horizontal
        assert count == 3  # Positions (10,5), (10,6), (10,7) for player 1
        
        # Test counting player -1 line
        count = game._count_line(12, 5, 0, 1, -1)  # Player -1, horizontal  
        assert count == 2  # Positions (12,5), (12,6) for player -1
    
    def test_get_board_state(self):
        """Test getting current board state."""
        game = GameState()
        
        # Empty board
        assert game.get_board_state() == []
        
        # Add some stones
        game.make_move((5, 5))
        game.make_move((10, 10))
        
        board_state = game.get_board_state()
        assert len(board_state) == 2
        assert (5, 5, 1) in board_state
        assert (10, 10, -1) in board_state
    
    def test_string_representation(self):
        """Test string representation of game state."""
        game = GameState()
        game.make_move((9, 9))
        
        game_str = str(game)
        assert "Move 1" in game_str
        assert "Player -1 to play" in game_str
        assert "Captures" in game_str
    
    def test_complex_capture_scenario(self):
        """Test complex capture scenario with multiple directions."""
        game = GameState()
        
        # Set up a position where a move can capture in multiple directions
        # This is complex to set up naturally, so we'll place stones directly
        
        # Place stones for potential captures
        game.board.set_stone(5, 5, 1)   # Center stone
        game.board.set_stone(5, 6, -1)  # Horizontal capture setup
        game.board.set_stone(5, 7, -1)
        game.board.set_stone(5, 8, 1)
        
        # Now if player 1 plays at a position that creates a capture...
        # Let's test the _check_captures method directly
        captured = game._check_captures(5, 8, 1)
        assert len(captured) == 2  # Should capture (5,6) and (5,7)
    
    def test_performance_single_state(self):
        """Test that game state modifications work on single instance."""
        game = GameState()
        original_board = game.board
        
        # Make several moves
        for i in range(10):
            game.make_move((i, i))
        
        # Board should be the same object (not copied)
        assert game.board is original_board
        
        # Undo all moves
        for i in range(10):
            game.undo_move()
        
        # Still the same board object
        assert game.board is original_board
        
        # Board should be empty again
        assert len(game.get_board_state()) == 0