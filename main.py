#!/usr/bin/env python3
"""
AlphaPente - High-Performance MCTS Pente Implementation
Entry point for running games, benchmarks, and analysis.
"""

import argparse
import time
import sys
from typing import Tuple, Optional, List

from src.core.game_state import GameState
from src.mcts.engine import MCTSEngine

class PenteGame:
    """Main game controller for Pente with MCTS AI."""
    
    def __init__(self, ai_time_ms: int = 1000, ai_max_sims: int = 1000):
        self.ai_engine = MCTSEngine(
            time_limit_ms=ai_time_ms,
            max_simulations=ai_max_sims
        )
        self.game_state = GameState()
        self.move_history: List[Tuple[str, Tuple[int, int], float]] = []  # (player, move, time)
        
    def reset_game(self):
        """Reset for a new game."""
        self.game_state = GameState()
        self.ai_engine.reset_tree()
        self.move_history = []
    
    def print_board(self):
        """Print the current board state."""
        print(f"\n{self.game_state}")
        print(f"Captures - Player 1: {self.game_state.captures[1]}, Player -1: {self.game_state.captures[-1]}")
    
    def get_human_move(self) -> Tuple[int, int]:
        """Get move input from human player."""
        while True:
            try:
                move_input = input(f"Enter your move (row,col) or 'quit': ").strip()
                if move_input.lower() == 'quit':
                    sys.exit(0)
                
                # Parse move
                parts = move_input.replace(' ', '').split(',')
                if len(parts) != 2:
                    print("Please enter move as: row,col (e.g., 9,9)")
                    continue
                
                row, col = int(parts[0]), int(parts[1])
                
                # Validate move
                if not (0 <= row < 19 and 0 <= col < 19):
                    print("Move must be within board (0-18, 0-18)")
                    continue
                
                if not self.game_state.board.is_empty(row, col):
                    print("Position is already occupied")
                    continue
                
                # Check tournament rule
                legal_moves = self.game_state.get_legal_moves()
                if (row, col) not in legal_moves:
                    print("Move violates tournament rule (second move must be ≥3 from center)")
                    continue
                
                return (row, col)
                
            except (ValueError, IndexError):
                print("Invalid input. Please enter move as: row,col (e.g., 9,9)")
            except KeyboardInterrupt:
                print("\nGoodbye!")
                sys.exit(0)
    
    def get_ai_move(self) -> Tuple[Tuple[int, int], dict]:
        """Get move from AI and return move with analysis."""
        start_time = time.time()
        
        # Get AI move
        move = self.ai_engine.search(self.game_state)
        
        end_time = time.time()
        move_time = end_time - start_time
        
        # Get detailed analysis
        analysis = self.ai_engine.get_move_analysis()
        stats = self.ai_engine.get_performance_stats()
        
        return move, {
            'move_time': move_time,
            'analysis': analysis,
            'stats': stats
        }
    
    def make_move(self, move: Tuple[int, int], player_name: str, move_time: float = 0.0):
        """Make a move and update game state."""
        self.game_state.make_move(move)
        self.move_history.append((player_name, move, move_time))
        
        # Update AI tree for tree reuse
        if hasattr(self, 'ai_engine'):
            self.ai_engine.update_root(move)
    
    def play_human_vs_ai(self, human_is_first: bool = True):
        """Play human vs AI game."""
        print("=== AlphaPente: Human vs AI ===")
        print("Board coordinates: (0,0) to (18,18)")
        print("Center is at (9,9)")
        print("Enter moves as: row,col (e.g., 9,9)")
        print("Type 'quit' to exit\n")
        
        self.reset_game()
        self.print_board()
        
        while not self.game_state.is_terminal():
            current_player = self.game_state.current_player
            
            if (human_is_first and current_player == 1) or (not human_is_first and current_player == -1):
                # Human turn
                print(f"\nYour turn (Player {'X' if current_player == 1 else 'O'}):")
                move = self.get_human_move()
                self.make_move(move, "Human")
                print(f"You played: {move}")
                
            else:
                # AI turn
                print(f"\nAI thinking (Player {'X' if current_player == 1 else 'O'})...")
                move, ai_data = self.get_ai_move()
                self.make_move(move, "AI", ai_data['move_time'])
                
                # Show AI analysis
                stats = ai_data['stats']
                print(f"AI played: {move}")
                print(f"AI stats: {stats['simulations']} simulations in {stats['time_ms']:.1f}ms "
                      f"({stats['simulations_per_second']:.1f} sims/sec)")
                
                if len(ai_data['analysis'].get('move_analysis', {})) > 1:
                    print("Top 3 AI moves:")
                    for i, (move_pos, data) in enumerate(list(ai_data['analysis']['move_analysis'].items())[:3]):
                        print(f"  {i+1}. {move_pos}: {data['visits']} visits ({data['win_rate']:.1f}% win rate)")
            
            self.print_board()
        
        # Game over
        winner = self.game_state.get_winner()
        if winner == 1:
            winner_name = "Player X" + (" (You)" if human_is_first else " (AI)")
        elif winner == -1:
            winner_name = "Player O" + (" (You)" if not human_is_first else " (AI)")
        else:
            winner_name = "Draw"
        
        print(f"\n🎉 Game Over! Winner: {winner_name}")
        self.print_game_summary()
    
    def play_ai_vs_ai(self, games: int = 1, show_moves: bool = True):
        """Play AI vs AI games for testing."""
        print(f"=== AlphaPente: AI vs AI ({games} game{'s' if games > 1 else ''}) ===")
        
        results = {'X_wins': 0, 'O_wins': 0, 'draws': 0}
        total_moves = 0
        total_time = 0
        
        for game_num in range(games):
            if games > 1:
                print(f"\nGame {game_num + 1}/{games}")
            
            self.reset_game()
            game_start_time = time.time()
            
            if show_moves and games == 1:
                self.print_board()
            
            while not self.game_state.is_terminal():
                current_player = self.game_state.current_player
                player_name = f"AI {'X' if current_player == 1 else 'O'}"
                
                move, ai_data = self.get_ai_move()
                self.make_move(move, player_name, ai_data['move_time'])
                
                if show_moves and games == 1:
                    stats = ai_data['stats']
                    print(f"\n{player_name} played: {move}")
                    print(f"Stats: {stats['simulations']} simulations, {stats['simulations_per_second']:.1f} sims/sec")
                    self.print_board()
            
            # Record result
            winner = self.game_state.get_winner()
            if winner == 1:
                results['X_wins'] += 1
            elif winner == -1:
                results['O_wins'] += 1
            else:
                results['draws'] += 1
            
            total_moves += self.game_state.move_count
            total_time += time.time() - game_start_time
            
            if show_moves and games == 1:
                winner_name = {1: "Player X", -1: "Player O", None: "Draw"}[winner]
                print(f"\n🎉 Game Over! Winner: {winner_name}")
        
        # Summary
        print(f"\n=== Results Summary ===")
        print(f"Player X wins: {results['X_wins']} ({results['X_wins']/games*100:.1f}%)")
        print(f"Player O wins: {results['O_wins']} ({results['O_wins']/games*100:.1f}%)")
        print(f"Draws: {results['draws']} ({results['draws']/games*100:.1f}%)")
        print(f"Average game length: {total_moves/games:.1f} moves")
        print(f"Average game time: {total_time/games:.1f} seconds")
        
        return results
    
    def print_game_summary(self):
        """Print summary of the completed game."""
        print(f"\n=== Game Summary ===")
        print(f"Total moves: {len(self.move_history)}")
        
        human_times = [t for player, move, t in self.move_history if player == "Human" and t > 0]
        ai_times = [t for player, move, t in self.move_history if player == "AI" and t > 0]
        
        if human_times:
            print(f"Average human move time: {sum(human_times)/len(human_times):.1f}s")
        if ai_times:
            print(f"Average AI move time: {sum(ai_times)/len(ai_times):.1f}s")
        
        print("Move history:")
        for i, (player, move, move_time) in enumerate(self.move_history):
            time_str = f" ({move_time:.1f}s)" if move_time > 0 else ""
            print(f"  {i+1}. {player}: {move}{time_str}")

def benchmark_performance(time_limits: List[int] = [100, 500, 1000], 
                         max_sims: List[int] = [50, 100, 500]):
    """Benchmark MCTS performance with different settings."""
    print("=== AlphaPente Performance Benchmark ===")
    
    game_state = GameState(tournament_rule=False)  # Simplify for consistent benchmarking
    
    print(f"{'Time Limit':<12} {'Max Sims':<10} {'Actual Sims':<12} {'Time (ms)':<12} {'Sims/sec':<12} {'Best Move'}")
    print("-" * 70)
    
    for time_limit in time_limits:
        for max_sim in max_sims:
            engine = MCTSEngine(time_limit_ms=time_limit, max_simulations=max_sim)
            
            # Run benchmark
            move = engine.search(game_state)
            stats = engine.get_performance_stats()
            
            print(f"{time_limit:<12} {max_sim:<10} {stats['simulations']:<12} "
                  f"{stats['time_ms']:<12.1f} {stats['simulations_per_second']:<12.1f} {move}")
    
    # Test tree reuse
    print(f"\n=== Tree Reuse Test ===")
    engine = MCTSEngine(time_limit_ms=200, max_simulations=100)
    game_state = GameState(tournament_rule=False)
    
    # First move
    move1 = engine.search(game_state)
    stats1 = engine.get_performance_stats()
    game_state.make_move(move1)
    
    # Update tree
    reused = engine.update_root(move1)
    
    # Second move
    move2 = engine.search(game_state)
    stats2 = engine.get_performance_stats()
    
    print(f"Move 1: {move1}, Tree size: {stats1.get('tree_size', 'N/A')}")
    print(f"Tree reused: {reused}")
    print(f"Move 2: {move2}, Tree size: {stats2.get('tree_size', 'N/A')}")

def analyze_position():
    """Interactive position analysis tool."""
    print("=== AlphaPente Position Analysis ===")
    print("Enter moves to build a position, then analyze with AI")
    print("Commands: 'move row,col', 'undo', 'analyze', 'reset', 'quit'")
    
    game_state = GameState()
    engine = MCTSEngine(time_limit_ms=2000, max_simulations=500)
    
    print(f"\n{game_state}")
    
    while True:
        try:
            command = input("\nEnter command: ").strip().lower()
            
            if command == 'quit':
                break
            elif command == 'reset':
                game_state = GameState()
                engine.reset_tree()
                print(f"\n{game_state}")
            elif command == 'undo':
                if game_state.move_history:
                    game_state.undo_move()
                    print(f"\n{game_state}")
                else:
                    print("No moves to undo")
            elif command == 'analyze':
                if game_state.is_terminal():
                    winner = game_state.get_winner()
                    winner_name = {1: "Player X", -1: "Player O", None: "Draw"}[winner]
                    print(f"Game is over. Winner: {winner_name}")
                else:
                    print("Analyzing position...")
                    move = engine.search(game_state)
                    analysis = engine.get_move_analysis()
                    stats = engine.get_performance_stats()
                    
                    print(f"\nAI Analysis:")
                    print(f"Best move: {move}")
                    print(f"Search stats: {stats['simulations']} sims in {stats['time_ms']:.1f}ms ({stats['simulations_per_second']:.1f}/sec)")
                    
                    if 'move_analysis' in analysis and analysis['move_analysis']:
                        print(f"\nTop moves:")
                        for i, (pos, data) in enumerate(list(analysis['move_analysis'].items())[:5]):
                            print(f"  {i+1}. {pos}: {data['visits']} visits, {data['win_rate']:.1f}% win rate")
                    
                    pv = engine.get_principal_variation(5)
                    if pv:
                        print(f"Principal variation: {' → '.join(str(move) for move in pv)}")
            elif command.startswith('move '):
                try:
                    move_str = command[5:].replace(' ', '')
                    row, col = map(int, move_str.split(','))
                    
                    if (row, col) not in game_state.get_legal_moves():
                        print("Illegal move")
                        continue
                    
                    game_state.make_move((row, col))
                    engine.update_root((row, col))
                    print(f"\n{game_state}")
                    
                except (ValueError, IndexError):
                    print("Invalid move format. Use: move row,col")
            else:
                print("Unknown command. Use: move row,col, undo, analyze, reset, quit")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description="AlphaPente - High-Performance MCTS Pente AI")
    parser.add_argument('mode', choices=['human', 'ai-vs-ai', 'benchmark', 'analyze'],
                       help='Game mode to run')
    parser.add_argument('--ai-time', type=int, default=1000,
                       help='AI thinking time in milliseconds (default: 1000)')
    parser.add_argument('--ai-sims', type=int, default=1000,
                       help='AI max simulations (default: 1000)')
    parser.add_argument('--games', type=int, default=1,
                       help='Number of AI vs AI games to play (default: 1)')
    parser.add_argument('--human-first', action='store_true',
                       help='Human plays first (default: AI first)')
    parser.add_argument('--show-moves', action='store_true',
                       help='Show individual moves in AI vs AI (default: only for single games)')
    parser.add_argument('--no-show-moves', action='store_true',
                       help='Hide individual moves in AI vs AI')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'human':
            game = PenteGame(ai_time_ms=args.ai_time, ai_max_sims=args.ai_sims)
            game.play_human_vs_ai(human_is_first=args.human_first)
            
        elif args.mode == 'ai-vs-ai':
            game = PenteGame(ai_time_ms=args.ai_time, ai_max_sims=args.ai_sims)
            # Show moves by default for single games, hide for multiple games
            show_moves = not args.no_show_moves and (args.show_moves or args.games == 1)
            game.play_ai_vs_ai(games=args.games, show_moves=show_moves)
            
        elif args.mode == 'benchmark':
            benchmark_performance()
            
        elif args.mode == 'analyze':
            analyze_position()
            
    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)

if __name__ == "__main__":
    main()