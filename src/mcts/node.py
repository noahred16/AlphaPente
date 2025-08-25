"""
MCTSNode class for Monte Carlo Tree Search.
Implements UCB1 selection and progressive widening.
"""

import math
from typing import Dict, List, Optional, Tuple
from ..core.game_state import GameState
from ..core.move_generator import get_move_generator

class MCTSNode:
    """
    Node in the MCTS tree.
    
    Key features:
    - UCB1 calculation for selection
    - Progressive widening based on visit count
    - Tracks wins/visits for backpropagation
    - Parent/child relationships for tree structure
    """
    
    def __init__(self, position: Optional[Tuple[int, int]] = None, parent: Optional['MCTSNode'] = None):
        """
        Initialize MCTS node.
        
        Args:
            position: The move that led to this node (None for root)
            parent: Parent node (None for root)
        """
        self.position = position
        self.parent = parent
        
        # MCTS statistics
        self.visits = 0
        self.wins = 0.0
        
        # Tree structure
        self.children: Dict[Tuple[int, int], 'MCTSNode'] = {}
        self.unexplored_moves: List[Tuple[int, int]] = []
        
        # Progressive widening
        self.available_moves: List[Tuple[int, int]] = []  # All possible moves
        self.widening_threshold = 0  # Number of moves currently allowed
        
        # Performance tracking
        self.is_terminal = False
        self.terminal_value: Optional[float] = None
    
    def is_fully_expanded(self) -> bool:
        """Check if all allowed moves have been explored."""
        return len(self.unexplored_moves) == 0
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf node (no children)."""
        return len(self.children) == 0
    
    def add_child(self, position: Tuple[int, int]) -> 'MCTSNode':
        """
        Add a child node for the given move.
        
        Args:
            position: Move position (row, col)
            
        Returns:
            The new child node
        """
        if position in self.children:
            raise ValueError(f"Child for position {position} already exists")
        
        child = MCTSNode(position, parent=self)
        self.children[position] = child
        
        # Remove from unexplored moves
        if position in self.unexplored_moves:
            self.unexplored_moves.remove(position)
        
        return child
    
    def update_available_moves(self, game_state: GameState) -> None:
        """
        Update available moves using progressive widening.
        
        Progressive widening thresholds:
        - 0 visits: distance 1, ~15 moves (initial expansion)
        - <10 visits: distance 1, ~15 moves
        - <100 visits: distance 2, ~30 moves  
        - <1000 visits: distance 3, ~50 moves
        - 1000+ visits: distance 5, ~80 moves
        """
        move_generator = get_move_generator()
        
        # Get moves based on progressive widening
        # Use at least 1 visit for initial expansion
        visit_count = max(self.visits, 1)
        new_moves = move_generator.get_progressive_widening_moves(game_state, visit_count)
        
        # Add any new moves to unexplored
        for move in new_moves:
            if move not in self.children and move not in self.unexplored_moves:
                self.unexplored_moves.append(move)
        
        self.available_moves = new_moves
        self.widening_threshold = len(new_moves)
    
    def select_child_ucb1(self, exploration_param: float = 1.414) -> 'MCTSNode':
        """
        Select child using UCB1 algorithm.
        
        UCB1 = win_rate + exploration_param * sqrt(ln(parent_visits) / child_visits)
        
        Args:
            exploration_param: Exploration parameter (sqrt(2) is theoretical optimum)
            
        Returns:
            Selected child node
        """
        if not self.children:
            raise ValueError("Cannot select child from node with no children")
        
        best_value = float('-inf')
        best_child = None
        
        for child in self.children.values():
            if child.visits == 0:
                # Unvisited child gets infinite UCB1 value
                return child
            
            # Calculate UCB1 value
            win_rate = child.wins / child.visits
            exploration = exploration_param * math.sqrt(math.log(self.visits) / child.visits)
            ucb1_value = win_rate + exploration
            
            if ucb1_value > best_value:
                best_value = ucb1_value
                best_child = child
        
        if best_child is None:
            raise ValueError("Failed to select child - no valid children found")
        
        return best_child
    
    def expand(self, game_state: GameState) -> Optional['MCTSNode']:
        """
        Expand the node by adding a child for an unexplored move.
        
        Args:
            game_state: Current game state
            
        Returns:
            New child node, or None if no moves to expand
        """
        # Update available moves based on current visit count
        self.update_available_moves(game_state)
        
        if not self.unexplored_moves:
            return None
        
        # Select first unexplored move (they're ordered by distance)
        move = self.unexplored_moves[0]
        return self.add_child(move)
    
    def backpropagate(self, result: float) -> None:
        """
        Backpropagate simulation result up the tree.
        
        Args:
            result: Simulation result (1.0 for win, 0.0 for loss, 0.5 for draw)
        """
        self.visits += 1
        self.wins += result
        
        if self.parent is not None:
            # Flip result for parent (opponent's perspective)
            self.parent.backpropagate(1.0 - result)
    
    def get_best_child(self, exploration_param: float = 0.0) -> Optional['MCTSNode']:
        """
        Get the best child for final move selection.
        
        Args:
            exploration_param: 0.0 for exploitation only (final selection)
            
        Returns:
            Best child node or None if no children
        """
        if not self.children:
            return None
        
        if exploration_param == 0.0:
            # Pure exploitation - highest win rate
            best_child = max(self.children.values(), key=lambda c: c.wins / max(c.visits, 1))
        else:
            # UCB1 selection
            best_child = self.select_child_ucb1(exploration_param)
        
        return best_child
    
    def get_most_visited_child(self) -> Optional['MCTSNode']:
        """Get child with most visits (robust selection)."""
        if not self.children:
            return None
        return max(self.children.values(), key=lambda c: c.visits)
    
    def get_visit_distribution(self) -> Dict[Tuple[int, int], int]:
        """Get visit count distribution for children."""
        return {pos: child.visits for pos, child in self.children.items()}
    
    def get_win_rates(self) -> Dict[Tuple[int, int], float]:
        """Get win rate distribution for children."""
        return {
            pos: child.wins / max(child.visits, 1) 
            for pos, child in self.children.items()
        }
    
    def prune_subtree(self, keep_child: 'MCTSNode') -> None:
        """
        Prune all children except the specified one (for tree reuse).
        
        Args:
            keep_child: Child to keep as the new subtree
        """
        # Remove all other children (garbage collection will clean up)
        new_children = {}
        for pos, child in self.children.items():
            if child is keep_child:
                new_children[pos] = child
                child.parent = None  # This becomes the new root
        
        self.children = new_children
    
    def get_statistics(self) -> Dict:
        """Get node statistics for debugging."""
        return {
            'position': self.position,
            'visits': self.visits,
            'wins': self.wins,
            'win_rate': self.wins / max(self.visits, 1),
            'children_count': len(self.children),
            'unexplored_count': len(self.unexplored_moves),
            'available_moves': len(self.available_moves),
            'widening_threshold': self.widening_threshold,
            'is_terminal': self.is_terminal,
            'is_fully_expanded': self.is_fully_expanded(),
            'is_leaf': self.is_leaf()
        }
    
    def __str__(self) -> str:
        """String representation for debugging."""
        win_rate = self.wins / max(self.visits, 1)
        return f"MCTSNode({self.position}, visits={self.visits}, win_rate={win_rate:.3f}, children={len(self.children)})"
    
    def __repr__(self) -> str:
        return self.__str__()