import math
import random
from typing import List, Tuple, Optional
from src.games.pente import Pente
from .move_heuristic import MoveHeuristic

class MCTSNode:
    """Node in the MCTS tree."""
    
    def __init__(self, game_state: Pente, move: Optional[Tuple[int, int]] = None, parent: Optional['MCTSNode'] = None):
        self.game_state = game_state.clone()
        self.move = move
        self.parent = parent
        self.children: List['MCTSNode'] = []
        self.visits = 0
        self.wins = 0.0
        self.untried_moves = game_state.get_legal_moves()
        self._cached_heuristic = None  # Cache heuristic instance
        
        # Sort moves by heuristic score for better ordering
        if self.untried_moves:
            self._cached_heuristic = MoveHeuristic(game_state)
            scored_moves = self._cached_heuristic.evaluate_moves(self.untried_moves)
            self.untried_moves = [move for move, score in scored_moves]
    
    def is_fully_expanded(self) -> bool:
        """Check if all legal moves have been tried."""
        return len(self.untried_moves) == 0
    
    def is_terminal(self) -> bool:
        """Check if this is a terminal node."""
        return self.game_state.is_terminal()
    
    def ucb1_score(self, exploration_weight: float = 1.414) -> float:
        """Calculate UCB1 score for node selection."""
        if self.visits == 0:
            return float('inf')
        
        # Normalize scaled values to [0,1] range for UCB1 calculation
        # Scaled values range from [-1, 1], so we map to [0, 1]
        avg_scaled_value = self.wins / self.visits
        exploitation = (avg_scaled_value + 1.0) / 2.0  # Map [-1,1] to [0,1]
        
        exploration = exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration
    
    def best_child(self, exploration_weight: float = 1.414) -> 'MCTSNode':
        """Select best child using UCB1."""
        return max(self.children, key=lambda child: child.ucb1_score(exploration_weight))
    
    def add_child(self, move: Tuple[int, int]) -> 'MCTSNode':
        """Add a child node for the given move."""
        new_game_state = self.game_state.clone()
        new_game_state.make_move(move)
        child = MCTSNode(new_game_state, move, self)
        self.children.append(child)
        self.untried_moves.remove(move)
        return child
    
    def update(self, result: float) -> None:
        """Update node statistics with simulation result."""
        self.visits += 1
        self.wins += result
    
    def backpropagate(self, result: float) -> None:
        """Backpropagate result up the tree."""
        self.update(result)
        if self.parent:
            # Flip result for opponent (negate scaled values)
            self.parent.backpropagate(-result)

class MCTS:
    """Monte Carlo Tree Search with move ordering heuristics."""
    
    def __init__(self, exploration_weight: float = 1.414, max_iterations: int = 1000):
        self.exploration_weight = exploration_weight
        self.max_iterations = max_iterations
        self._heuristic = None  # Shared heuristic instance
    
    def search(self, game_state: Pente) -> Tuple[int, int]:
        """Run MCTS and return the best move."""
        root = MCTSNode(game_state)
        root_player = game_state.current_player  # Store root player for consistent evaluation
        
        for _ in range(self.max_iterations):
            # Selection and Expansion
            leaf = self._select_and_expand(root)
            
            # Simulation with consistent perspective from root player
            result = self._simulate(leaf.game_state, root_player)
            
            # Backpropagation
            leaf.backpropagate(result)
        
        # Return move with best combination of visits and win rate
        if not root.children:
            # Fallback to heuristic ordering if no expansions
            legal_moves = game_state.get_legal_moves()
            if legal_moves:
                heuristic = MoveHeuristic(game_state)
                scored_moves = heuristic.evaluate_moves(legal_moves)
                return scored_moves[0][0]
            return (0, 0)  # Should not happen
        
        # Special case: Check for critical moves (win/block) before relying on MCTS statistics
        if self._heuristic is None:
            self._heuristic = MoveHeuristic(game_state)
        else:
            self._heuristic.game = game_state  # Update reference
        
        # First check all legal moves for immediate wins
        legal_moves = game_state.get_legal_moves()
        for move in legal_moves:
            move_score = self._heuristic._evaluate_move(move)
            if move_score >= 100:  # Critical move (win/block)
                # Check if this move is actually winning for current player
                if self._heuristic._is_winning_move(move, game_state.current_player):
                    return move
        
        # Then check for blocking moves
        for move in legal_moves:
            move_score = self._heuristic._evaluate_move(move)
            if move_score >= 100:  # Critical move (win/block)
                # Check if this move blocks opponent win
                if self._heuristic._is_blocking_win(move, -game_state.current_player):
                    return move
        
        # Sort by average scaled value (higher is better for root player)
        def move_priority(child):
            if child.visits == 0:
                return (0.0, 0)
            
            # Average scaled value from child's perspective
            child_avg_value = child.wins / child.visits
            # Convert to root player's perspective (negate since child represents opponent's move)
            root_perspective_value = -child_avg_value
            
            # Prioritize higher values, but require minimum visits for reliability  
            min_visits = max(1, self.max_iterations // 100)  # At least 1% of iterations (more lenient)
            if child.visits >= min_visits:
                return (root_perspective_value, child.visits)
            else:
                return (-1.0, child.visits)  # Low priority for insufficient data
        
        best_child = max(root.children, key=move_priority)
        return best_child.move
    
    def _select_and_expand(self, node: MCTSNode) -> MCTSNode:
        """Select a leaf node and expand it if possible."""
        current = node
        
        # Selection phase - traverse down to a leaf
        while not current.is_terminal() and current.is_fully_expanded():
            current = current.best_child(self.exploration_weight)
        
        # Expansion phase - add a new child if possible
        if not current.is_terminal() and not current.is_fully_expanded():
            # Select the best untried move (already ordered by heuristic)
            move = current.untried_moves[0]
            current = current.add_child(move)
        
        return current
    
    def _simulate(self, game_state: Pente, root_player: int = None) -> float:
        """Run a random simulation from the given game state using cloning for safety."""
        # Revert to cloning approach for safety - optimization wasn't worth the complexity
        simulation_game = game_state.clone()
        evaluation_player = root_player if root_player is not None else simulation_game.current_player
        
        while not simulation_game.is_terminal():
            legal_moves = simulation_game.get_legal_moves()
            if not legal_moves:
                break
            
            # Use epsilon-greedy selection for better performance
            if random.random() < 0.2:  # 20% random exploration
                selected_move = random.choice(legal_moves)
            else:
                # Quick heuristic-based selection (limit evaluation to top moves)
                if len(legal_moves) > 10:
                    # For large move sets, sample top candidates using simple heuristics
                    center = simulation_game.board_size // 2
                    def distance_score(move):
                        r, c = move
                        return -(abs(r - center) + abs(c - center))  # Prefer center moves
                    
                    legal_moves.sort(key=distance_score, reverse=True)
                    candidate_moves = legal_moves[:10]
                else:
                    candidate_moves = legal_moves
                
                # Create fresh heuristic for this evaluation
                heuristic = MoveHeuristic(simulation_game)
                scored_moves = heuristic.evaluate_moves(candidate_moves)
                selected_move = scored_moves[0][0] if scored_moves else random.choice(legal_moves)
            
            simulation_game.make_move(selected_move)
        
        # Return result from perspective of the evaluation player with move-count scaling
        winner = simulation_game.get_winner()
        move_count = simulation_game.move_count
        
        return self._calculate_scaled_valuation(winner, evaluation_player, move_count)
    
    def get_move_statistics(self, game_state: Pente) -> List[Tuple[Tuple[int, int], int, float]]:
        """Get detailed statistics for all explored moves."""
        root = MCTSNode(game_state)
        root_player = game_state.current_player
        
        for _ in range(self.max_iterations):
            leaf = self._select_and_expand(root)
            result = self._simulate(leaf.game_state, root_player)
            leaf.backpropagate(result)
        
        statistics = []
        for child in root.children:
            if child.visits > 0:
                child_avg_value = child.wins / child.visits
                # Convert to root player's perspective (negate since child represents opponent's move)
                root_perspective_value = -child_avg_value
            else:
                root_perspective_value = 0.0
            statistics.append((child.move, child.visits, root_perspective_value))
        
        return sorted(statistics, key=lambda x: x[1], reverse=True)
    
    def _calculate_scaled_valuation(self, winner, original_player, move_count):
        """
        Calculate scaled valuation that favors faster wins and slower losses.
        
        Args:
            winner: The winning player (1, -1, or None for tie)
            original_player: The player from whose perspective to evaluate
            move_count: Number of moves played in the game
            
        Returns:
            Scaled valuation:
            - Ties: 0.0
            - Wins: 0.2 to 1.0 (1.0 for fast wins, 0.2 for slow wins)
            - Losses: -0.2 to -1.0 (-1.0 for fast losses, -0.2 for slow losses)
        """
        if winner is None:
            return 0.0  # Tie is always neutral
        
        # Calculate scaling factor that decreases with more moves
        # Fast games (low move_count) get high scaling, slow games get low scaling
        scale_factor = 0.2 + 0.8 * math.exp(-move_count / 50.0)
        
        if winner == original_player:
            return scale_factor  # Win: positive value, higher for faster wins
        else:
            return -scale_factor  # Loss: negative value, more negative for faster losses