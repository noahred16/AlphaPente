import math
import random
from typing import List, Tuple, Optional
from games.pente import Pente
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
        self.heuristic = MoveHeuristic(game_state)
        
        # Sort moves by heuristic score for better ordering
        if self.untried_moves:
            scored_moves = self.heuristic.evaluate_moves(self.untried_moves)
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
        
        exploitation = self.wins / self.visits
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
            # Flip result for opponent
            self.parent.backpropagate(1.0 - result)

class MCTS:
    """Monte Carlo Tree Search with move ordering heuristics."""
    
    def __init__(self, exploration_weight: float = 1.414, max_iterations: int = 1000):
        self.exploration_weight = exploration_weight
        self.max_iterations = max_iterations
    
    def search(self, game_state: Pente) -> Tuple[int, int]:
        """Run MCTS and return the best move."""
        root = MCTSNode(game_state)
        
        for _ in range(self.max_iterations):
            # Selection and Expansion
            leaf = self._select_and_expand(root)
            
            # Simulation
            result = self._simulate(leaf.game_state)
            
            # Backpropagation
            leaf.backpropagate(result)
        
        # Return move with highest visit count
        if not root.children:
            # Fallback to heuristic ordering if no expansions
            legal_moves = game_state.get_legal_moves()
            if legal_moves:
                heuristic = MoveHeuristic(game_state)
                scored_moves = heuristic.evaluate_moves(legal_moves)
                return scored_moves[0][0]
            return (0, 0)  # Should not happen
        
        best_child = max(root.children, key=lambda child: child.visits)
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
    
    def _simulate(self, game_state: Pente) -> float:
        """Run a random simulation from the given game state."""
        simulation_game = game_state.clone()
        
        while not simulation_game.is_terminal():
            legal_moves = simulation_game.get_legal_moves()
            if not legal_moves:
                break
            
            # Use heuristic-guided random selection for better simulations
            heuristic = MoveHeuristic(simulation_game)
            scored_moves = heuristic.evaluate_moves(legal_moves)
            
            # Weighted random selection based on heuristic scores
            total_weight = sum(score for _, score in scored_moves)
            if total_weight > 0:
                rand_val = random.random() * total_weight
                cumulative = 0.0
                selected_move = scored_moves[0][0]  # fallback
                
                for move, score in scored_moves:
                    cumulative += score
                    if rand_val <= cumulative:
                        selected_move = move
                        break
            else:
                selected_move = random.choice(legal_moves)
            
            simulation_game.make_move(selected_move)
        
        # Return result from perspective of the player who made the root move
        winner = simulation_game.get_winner()
        if winner is None:
            return 0.5  # Draw
        elif winner == game_state.current_player:
            return 1.0  # Win
        else:
            return 0.0  # Loss
    
    def get_move_statistics(self, game_state: Pente) -> List[Tuple[Tuple[int, int], int, float]]:
        """Get detailed statistics for all explored moves."""
        root = MCTSNode(game_state)
        
        for _ in range(self.max_iterations):
            leaf = self._select_and_expand(root)
            result = self._simulate(leaf.game_state)
            leaf.backpropagate(result)
        
        statistics = []
        for child in root.children:
            win_rate = child.wins / child.visits if child.visits > 0 else 0.0
            statistics.append((child.move, child.visits, win_rate))
        
        return sorted(statistics, key=lambda x: x[1], reverse=True)