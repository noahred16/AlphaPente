"""Skill Level Evaluation System for AlphaPente Model Strength Assessment

This module defines different opponent skill levels and evaluation benchmarks to measure
model improvement over time with smooth, continuous metrics rather than binary pass/fail tests.

The evaluation system provides multiple difficulty tiers from random play to expert-level
tactical understanding, allowing fine-grained tracking of learning progress.

Key Concepts:
- Progressive skill levels create smooth learning curves
- Multiple evaluation dimensions capture different aspects of strength
- Statistical significance through multiple games per skill level
- Reproducible benchmarks for consistent measurement across training cycles

Skill Levels (Easy → Hard):
1. Random Player: Completely random legal moves
2. Greedy Player: Always plays highest-value immediate move
3. Tactical Player: Can block obvious threats and create basic attacks
4. MCTS-Weak: Low-simulation MCTS (50 simulations)
5. MCTS-Medium: Medium-simulation MCTS (200 simulations) 
6. MCTS-Strong: High-simulation MCTS (800 simulations)
7. Previous-Self: Model from N training cycles ago

Each level provides win rate, average game length, and tactical accuracy metrics.
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

from game import reset_game, get_legal_moves, make_move, is_game_over, calculate_score
from mcts import MCTS
from settings_loader import BOARD_SIZE, CONNECT_N, CAPTURES_ENABLED, TOURNAMENT_RULES_ENABLED


@dataclass
class GameResult:
    """Result of a single evaluation game."""
    winner: int  # 1 for model win, -1 for opponent win, 0 for tie
    game_length: int  # Number of moves in the game
    model_tactical_score: float  # 0-1 score for tactical play quality
    opponent_tactical_score: float  # 0-1 score for opponent play quality
    final_score: float  # Game result from model's perspective


@dataclass
class SkillLevelResult:
    """Aggregate results against a specific skill level."""
    skill_level: str
    games_played: int
    win_rate: float  # Percentage of games won
    avg_game_length: float  # Average moves per game
    avg_tactical_score: float  # Average tactical play quality
    score_std: float  # Standard deviation of tactical scores
    confidence_interval: Tuple[float, float]  # 95% CI for win rate


class OpponentPlayer(ABC):
    """Abstract base class for opponent players of different skill levels."""
    
    @abstractmethod
    def get_move(self, board: np.ndarray, player_captures: int, 
                 opponent_captures: int, num_moves: int) -> Tuple[int, int]:
        """Return the chosen move for this opponent type."""
        pass
    
    @abstractmethod
    def get_skill_level(self) -> str:
        """Return string identifier for this skill level."""
        pass


class RandomPlayer(OpponentPlayer):
    """Completely random legal moves."""
    
    def get_move(self, board: np.ndarray, player_captures: int, 
                 opponent_captures: int, num_moves: int) -> Tuple[int, int]:
        legal_moves = get_legal_moves(board, num_moves, TOURNAMENT_RULES_ENABLED)
        return random.choice(legal_moves)
    
    def get_skill_level(self) -> str:
        return "random"


class GreedyPlayer(OpponentPlayer):
    """Always plays the center-most legal move (simple heuristic)."""
    
    def get_move(self, board: np.ndarray, player_captures: int, 
                 opponent_captures: int, num_moves: int) -> Tuple[int, int]:
        legal_moves = get_legal_moves(board, num_moves, TOURNAMENT_RULES_ENABLED)
        
        # Prefer moves closer to center
        center = (BOARD_SIZE[0] // 2, BOARD_SIZE[1] // 2)
        
        def distance_to_center(move):
            return abs(move[0] - center[0]) + abs(move[1] - center[1])
        
        return min(legal_moves, key=distance_to_center)
    
    def get_skill_level(self) -> str:
        return "greedy"


class TacticalPlayer(OpponentPlayer):
    """Can detect and block immediate threats, create simple attacks."""
    
    def get_move(self, board: np.ndarray, player_captures: int, 
                 opponent_captures: int, num_moves: int) -> Tuple[int, int]:
        legal_moves = get_legal_moves(board, num_moves, TOURNAMENT_RULES_ENABLED)
        
        # Check for immediate wins
        for move in legal_moves:
            test_board = board.copy()
            make_move(move, test_board)
            if is_game_over(move, test_board, player_captures, CONNECT_N, num_moves + 1):
                return move
        
        # Check for blocks (opponent wins)
        test_board = board * -1  # Flip perspective
        for move in legal_moves:
            test_board_copy = test_board.copy()
            make_move(move, test_board_copy)
            if is_game_over(move, test_board_copy, opponent_captures, CONNECT_N, num_moves + 1):
                return move
        
        # Fallback to greedy
        center = (BOARD_SIZE[0] // 2, BOARD_SIZE[1] // 2)
        def distance_to_center(move):
            return abs(move[0] - center[0]) + abs(move[1] - center[1])
        
        return min(legal_moves, key=distance_to_center)
    
    def get_skill_level(self) -> str:
        return "tactical"


class MCTSPlayer(OpponentPlayer):
    """MCTS-based player with configurable simulation count."""
    
    def __init__(self, model, simulations: int):
        self.model = model
        self.simulations = simulations
        self.mcts = MCTS(model, simulations=simulations, random=0.1)
        self.level_name = f"mcts_{simulations}"
    
    def get_move(self, board: np.ndarray, player_captures: int, 
                 opponent_captures: int, num_moves: int) -> Tuple[int, int]:
        move, _ = self.mcts.best_move(board, player_captures, opponent_captures)
        return move
    
    def get_skill_level(self) -> str:
        return self.level_name


def calculate_tactical_score(board: np.ndarray, move: Tuple[int, int], 
                           player_captures: int, opponent_captures: int) -> float:
    """
    Calculate a tactical score (0-1) for how good a move is.
    
    This provides a continuous metric instead of binary win/loss.
    Higher scores indicate better tactical understanding.
    """
    if move is None:
        return 0.0
    
    score = 0.0
    
    # Base score for making a legal move
    score += 0.2
    
    # Bonus for center play early in game
    center = (BOARD_SIZE[0] // 2, BOARD_SIZE[1] // 2)
    distance_to_center = abs(move[0] - center[0]) + abs(move[1] - center[1])
    if np.sum(np.abs(board)) < 10:  # Early game
        score += 0.3 * (1.0 - distance_to_center / 6.0)
    
    # Check if move creates threats or blocks threats
    test_board = board.copy()
    make_move(move, test_board)
    
    # Bonus for creating winning threats
    if is_game_over(move, test_board, player_captures, CONNECT_N, np.sum(np.abs(board)) + 1):
        score += 0.5  # Winning move
    
    # Bonus for blocking opponent wins
    opp_board = board * -1
    legal_moves = get_legal_moves(opp_board, np.sum(np.abs(board)), TOURNAMENT_RULES_ENABLED)
    opponent_can_win = False
    
    for opp_move in legal_moves:
        opp_test_board = opp_board.copy()
        make_move(opp_move, opp_test_board)
        if is_game_over(opp_move, opp_test_board, opponent_captures, CONNECT_N, np.sum(np.abs(board)) + 1):
            if opp_move == move:  # Our move blocks their win
                score += 0.4
            else:
                opponent_can_win = True
    
    # Penalty for not blocking when opponent can win
    if opponent_can_win:
        score -= 0.3
    
    return min(1.0, max(0.0, score))


def create_skill_levels(model) -> List[OpponentPlayer]:
    """Create the range of skill levels for evaluation (optimized for speed)."""
    return [
        RandomPlayer(),
        GreedyPlayer(), 
        TacticalPlayer(),
        MCTSPlayer(model, simulations=50),
        MCTSPlayer(model, simulations=200),
        # Removed MCTS-500 and MCTS-800 for faster evaluation during development
    ]


def play_evaluation_game(model, opponent: OpponentPlayer, 
                        model_plays_first: bool = True) -> GameResult:
    """
    Play a single evaluation game between model and opponent.
    
    Returns detailed game statistics for smooth metrics.
    """
    board, player_captures, opponent_captures = reset_game(BOARD_SIZE)
    num_moves = 0
    model_tactical_scores = []
    opponent_tactical_scores = []
    
    # Create MCTS for the model
    model_mcts = MCTS(model, simulations=300, random=0.05)
    
    result = None
    while result is None:
        current_player_is_model = (num_moves % 2 == 0) == model_plays_first
        
        if current_player_is_model:
            # Model's turn
            move, _ = model_mcts.best_move(board, player_captures, opponent_captures)
            tactical_score = calculate_tactical_score(board, move, player_captures, opponent_captures)
            model_tactical_scores.append(tactical_score)
        else:
            # Opponent's turn
            move = opponent.get_move(board, player_captures, opponent_captures, num_moves)
            tactical_score = calculate_tactical_score(board, move, player_captures, opponent_captures)
            opponent_tactical_scores.append(tactical_score)
        
        # Make the move
        make_move(move, board)
        num_moves += 1
        
        # Check for game end
        if is_game_over(move, board, player_captures, CONNECT_N, num_moves):
            raw_result = calculate_score(num_moves, BOARD_SIZE, CAPTURES_ENABLED)
            # Adjust result based on who's playing first
            if model_plays_first:
                final_result = raw_result if num_moves % 2 == 1 else -raw_result
            else:
                final_result = -raw_result if num_moves % 2 == 1 else raw_result
            break
        
        # Flip board for next player
        board = board * -1
        player_captures, opponent_captures = opponent_captures, player_captures
    
    # Clean up
    model_mcts.clear_tree()
    
    # Determine winner from model's perspective
    if final_result > 0:
        winner = 1   # Model wins
    elif final_result < 0:
        winner = -1  # Opponent wins
    else:
        winner = 0   # Tie
    
    return GameResult(
        winner=winner,
        game_length=num_moves,
        model_tactical_score=np.mean(model_tactical_scores) if model_tactical_scores else 0.0,
        opponent_tactical_score=np.mean(opponent_tactical_scores) if opponent_tactical_scores else 0.0,
        final_score=final_result
    )