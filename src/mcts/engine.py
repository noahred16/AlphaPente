"""
MCTS Engine - The core Monte Carlo Tree Search implementation.
This is the heart of the high-performance Pente AI system.
"""

import time
import math
from typing import Optional, Tuple, List, Dict
from ..core.game_state import GameState
from .node import MCTSNode
from .rollout import RolloutPolicy

class MCTSEngine:
    """
    High-performance MCTS engine for Pente.
    
    Key features from specs:
    - Single GameState (never copied!) with make/undo
    - Tree reuse via update_root()
    - Progressive widening in tree policy
    - Target: 10,000+ simulations per second
    - Preserve 50-80% of tree between moves
    """
    
    def __init__(self, 
                 exploration_param: float = 1.414,
                 rollout_max_moves: int = 200,
                 time_limit_ms: int = 1000,
                 max_simulations: int = 10000,
                 early_termination_threshold: float = 3.0):
        """
        Initialize MCTS engine.
        
        Args:
            exploration_param: UCB1 exploration parameter (sqrt(2) = 1.414 is theoretical optimum)
            rollout_max_moves: Maximum moves in rollout policy
            time_limit_ms: Time limit per search in milliseconds
            max_simulations: Maximum simulations per search
            early_termination_threshold: Stop when best move has this many times more visits than second-best
        """
        self.exploration_param = exploration_param
        self.time_limit_ms = time_limit_ms
        self.max_simulations = max_simulations
        self.early_termination_threshold = early_termination_threshold
        
        # MCTS components
        self.rollout_policy = RolloutPolicy(max_moves=rollout_max_moves)
        
        # Tree state
        self.root: Optional[MCTSNode] = None
        self.game_state: Optional[GameState] = None
        
        # Performance tracking
        self.last_search_stats = {}
        
    def initialize_search(self, game_state: GameState) -> None:
        """
        Initialize search with a game state.
        
        CRITICAL: This takes ownership of the game state - it will be modified!
        
        Args:
            game_state: Game state to search from (will be modified during search)
        """
        self.game_state = game_state
        
        # Create root node if needed
        if self.root is None:
            self.root = MCTSNode()
            
        # Update root's available moves based on current state
        self.root.update_available_moves(game_state)
    
    def search(self, game_state: GameState, time_limit_ms: Optional[int] = None) -> Tuple[int, int]:
        """
        Perform MCTS search and return the best move.
        
        Args:
            game_state: Current game state (will be modified during search then restored)
            time_limit_ms: Override default time limit
            
        Returns:
            Best move as (row, col) tuple
            
        Raises:
            ValueError: If no legal moves available
        """
        # Initialize search
        self.initialize_search(game_state)
        
        # Check for legal moves
        legal_moves = game_state.get_legal_moves()
        if not legal_moves:
            raise ValueError("No legal moves available")
            
        # If only one move, return it immediately
        if len(legal_moves) == 1:
            self.last_search_stats = {
                'simulations': 0,
                'time_ms': 0,
                'simulations_per_second': 0,
                'best_move': legal_moves[0],
                'best_move_visits': 0,
                'total_children': 1,
                'early_termination': True
            }
            return legal_moves[0]
        
        # Search parameters
        time_limit = time_limit_ms if time_limit_ms is not None else self.time_limit_ms
        start_time = time.time()
        simulations = 0
        
        # Main MCTS loop
        while simulations < self.max_simulations:
            # Check time limit
            current_time = time.time()
            elapsed_ms = (current_time - start_time) * 1000
            if elapsed_ms >= time_limit:
                break
            
            # Perform one simulation
            self._simulate()
            simulations += 1
            
            # Check for early termination every 100 simulations
            if simulations % 100 == 0 and simulations > 200:
                if self._should_terminate_early():
                    break
        
        # Get best move
        best_child = self.root.get_most_visited_child()
        if best_child is None:
            # Fallback to first legal move if no children expanded
            best_move = legal_moves[0]
            best_visits = 0
        else:
            best_move = best_child.position
            best_visits = best_child.visits
        
        # Calculate performance stats
        end_time = time.time()
        total_time_ms = (end_time - start_time) * 1000
        sims_per_second = (simulations / max(total_time_ms / 1000, 0.001))
        
        self.last_search_stats = {
            'simulations': simulations,
            'time_ms': total_time_ms,
            'simulations_per_second': sims_per_second,
            'best_move': best_move,
            'best_move_visits': best_visits,
            'total_children': len(self.root.children),
            'tree_size': self._count_tree_nodes(self.root),
            'early_termination': simulations < self.max_simulations
        }
        
        return best_move
    
    def _simulate(self) -> None:
        """
        Perform one MCTS simulation: Selection -> Expansion -> Rollout -> Backpropagation.
        
        This is the critical path for performance - must be as fast as possible.
        """
        # Track moves made during simulation for undo
        moves_made: List[Tuple[int, int]] = []
        
        try:
            # 1. SELECTION: Navigate to leaf using tree policy
            node = self._select_leaf(moves_made)
            
            # 2. EXPANSION: Add child if not terminal
            if not self.game_state.is_terminal() and not node.is_fully_expanded():
                child = node.expand(self.game_state)
                if child is not None:
                    # Make the child's move
                    self.game_state.make_move(child.position)
                    moves_made.append(child.position)
                    node = child
            
            # 3. ROLLOUT: Simulate random game to completion
            result = self.rollout_policy.rollout(self.game_state)
            
            # Convert result to current player's perspective
            # (rollout returns result from game_state.current_player's perspective)
            current_player = self.game_state.current_player
            original_player = 1 if (len(moves_made) % 2) == 0 else -1
            
            if current_player == original_player:
                final_result = result
            else:
                final_result = 1.0 - result
            
            # 4. BACKPROPAGATION: Update statistics up the tree
            node.backpropagate(final_result)
            
        finally:
            # CRITICAL: Undo all moves to restore original state
            for _ in range(len(moves_made)):
                self.game_state.undo_move()
    
    def _select_leaf(self, moves_made: List[Tuple[int, int]]) -> MCTSNode:
        """
        Select leaf node using tree policy (UCB1 + progressive widening).
        
        Args:
            moves_made: List to track moves made (for undo)
            
        Returns:
            Leaf node to expand/rollout from
        """
        node = self.root
        
        # Navigate down tree using UCB1
        while not node.is_leaf() and not self.game_state.is_terminal():
            # Select best child using UCB1
            child = node.select_child_ucb1(self.exploration_param)
            
            # Make the move to reach child
            self.game_state.make_move(child.position)
            moves_made.append(child.position)
            
            node = child
        
        return node
    
    def _should_terminate_early(self) -> bool:
        """
        Check if search should terminate early due to clear best move.
        
        Returns:
            True if search should terminate early
        """
        if len(self.root.children) < 2:
            return False
        
        # Get visit counts for all children
        visit_counts = [child.visits for child in self.root.children.values()]
        visit_counts.sort(reverse=True)
        
        if len(visit_counts) < 2 or visit_counts[1] == 0:
            return False
        
        # Check if best move dominates
        ratio = visit_counts[0] / visit_counts[1]
        return ratio >= self.early_termination_threshold
    
    def update_root(self, move: Tuple[int, int]) -> bool:
        """
        Update root after opponent move for tree reuse.
        
        This is critical for performance - preserves 50-80% of tree between moves.
        
        Args:
            move: Move that was played
            
        Returns:
            True if tree was successfully reused, False if new tree created
        """
        if self.root is None or move not in self.root.children:
            # No tree to reuse or move not in tree
            self.root = MCTSNode()
            return False
        
        # Get child that corresponds to the move
        new_root = self.root.children[move]
        
        # Make child the new root (this also detaches other children for GC)
        new_root.parent = None
        self.root = new_root
        
        return True
    
    def get_move_analysis(self) -> Dict:
        """
        Get detailed analysis of the current tree state.
        
        Returns:
            Dictionary with move analysis data
        """
        if self.root is None or len(self.root.children) == 0:
            return {'error': 'No analysis available - no children in root'}
        
        # Get visit and win rate data for all children
        move_data = {}
        total_visits = sum(child.visits for child in self.root.children.values())
        
        for move, child in self.root.children.items():
            visit_percentage = (child.visits / max(total_visits, 1)) * 100
            win_rate = (child.wins / max(child.visits, 1)) * 100
            
            move_data[move] = {
                'visits': child.visits,
                'visit_percentage': visit_percentage,
                'wins': child.wins,
                'win_rate': win_rate,
                'ucb1_value': self._calculate_ucb1(child, self.root.visits)
            }
        
        # Sort by visits (most visited first)
        sorted_moves = sorted(move_data.items(), key=lambda x: x[1]['visits'], reverse=True)
        
        return {
            'total_root_visits': self.root.visits,
            'total_children': len(self.root.children),
            'move_analysis': dict(sorted_moves),
            'best_move': sorted_moves[0][0] if sorted_moves else None,
            'tree_depth': self._get_max_tree_depth(self.root),
            'last_search_stats': self.last_search_stats
        }
    
    def _calculate_ucb1(self, child: MCTSNode, parent_visits: int) -> float:
        """Calculate UCB1 value for a child node."""
        if child.visits == 0:
            return float('inf')
        
        win_rate = child.wins / child.visits
        exploration = self.exploration_param * math.sqrt(math.log(parent_visits) / child.visits)
        return win_rate + exploration
    
    def _count_tree_nodes(self, node: MCTSNode) -> int:
        """Count total nodes in subtree."""
        count = 1
        for child in node.children.values():
            count += self._count_tree_nodes(child)
        return count
    
    def _get_max_tree_depth(self, node: MCTSNode, current_depth: int = 0) -> int:
        """Get maximum depth of tree."""
        if not node.children:
            return current_depth
        
        max_depth = current_depth
        for child in node.children.values():
            child_depth = self._get_max_tree_depth(child, current_depth + 1)
            max_depth = max(max_depth, child_depth)
        
        return max_depth
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics from last search."""
        return self.last_search_stats.copy()
    
    def reset_tree(self) -> None:
        """Reset the search tree (start fresh)."""
        self.root = None
        self.game_state = None
        self.last_search_stats = {}
    
    def get_principal_variation(self, max_depth: int = 10) -> List[Tuple[int, int]]:
        """
        Get the principal variation (most visited path) from root.
        
        Args:
            max_depth: Maximum depth to traverse
            
        Returns:
            List of moves in the principal variation
        """
        if self.root is None:
            return []
        
        pv = []
        node = self.root
        
        for _ in range(max_depth):
            if not node.children:
                break
                
            # Get most visited child
            best_child = node.get_most_visited_child()
            if best_child is None:
                break
                
            pv.append(best_child.position)
            node = best_child
        
        return pv
    
    def __str__(self) -> str:
        """String representation for debugging."""
        if self.root is None:
            return "MCTSEngine(no tree)"
        
        stats = self.get_performance_stats()
        sims_per_sec = stats.get('simulations_per_second', 0)
        
        return (f"MCTSEngine(root_visits={self.root.visits}, "
                f"children={len(self.root.children)}, "
                f"last_perf={sims_per_sec:.0f} sims/sec)")