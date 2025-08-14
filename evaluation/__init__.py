"""AlphaPente Model Evaluation Framework

This package provides comprehensive evaluation tools for measuring model strength
and learning progress with smooth, continuous metrics.

Main Components:
- skill_levels: Defines opponent types from random to expert MCTS
- benchmark_suite: Complete evaluation framework with statistical analysis  
- performance_tracker: Integration with training pipeline for automated tracking

Quick Start:
    from evaluation import run_quick_benchmark, run_full_benchmark
    
    # Quick test (5 games per level)
    report = run_quick_benchmark(model, "my_model")
    
    # Full evaluation (50 games per level)  
    report = run_full_benchmark(model, "production_model", training_step=100)
    
    print(f"Model strength: {report.overall_strength_score:.1f}/100")
"""

from .skill_levels import (
    OpponentPlayer, RandomPlayer, GreedyPlayer, TacticalPlayer, MCTSPlayer,
    GameResult, SkillLevelResult, create_skill_levels, play_evaluation_game,
    calculate_tactical_score
)

from .benchmark_suite import (
    BenchmarkSuite, BenchmarkReport, run_quick_benchmark, run_full_benchmark
)

__all__ = [
    'OpponentPlayer', 'RandomPlayer', 'GreedyPlayer', 'TacticalPlayer', 'MCTSPlayer',
    'GameResult', 'SkillLevelResult', 'BenchmarkSuite', 'BenchmarkReport',
    'create_skill_levels', 'play_evaluation_game', 'calculate_tactical_score',
    'run_quick_benchmark', 'run_full_benchmark'
]