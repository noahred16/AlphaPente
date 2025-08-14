"""Comprehensive Benchmark Suite for Model Evaluation

This module provides a complete evaluation framework that generates smooth, continuous 
metrics for tracking model improvement over time. Unlike binary pass/fail tests,
these benchmarks provide detailed performance curves and statistical analysis.

Key Features:
- Multiple skill levels from random to expert MCTS
- Statistical significance testing with confidence intervals
- Continuous tactical scoring (0-1) for smooth learning curves
- Performance tracking across training cycles
- Detailed analysis reports with visualizations
- Integration with training pipeline for automated evaluation

Evaluation Dimensions:
1. Win rates against different skill levels (primary metric)
2. Tactical play quality scores (secondary metric) 
3. Game length analysis (efficiency metric)
4. Consistency measurements (reliability metric)
5. Improvement rate tracking (learning speed metric)

The system generates evaluation reports that can be compared across training
cycles to demonstrate clear learning progress with statistical confidence.
"""

import numpy as np
import random
import json
import os
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import statistics

from .skill_levels import (
    OpponentPlayer, SkillLevelResult, GameResult,
    create_skill_levels, play_evaluation_game, RandomPlayer
)


@dataclass 
class BenchmarkReport:
    """Complete evaluation report for a model at a specific training stage."""
    model_name: str
    training_step: int
    timestamp: str
    skill_level_results: List[SkillLevelResult]
    overall_strength_score: float  # 0-100 composite strength metric
    tactical_ability_score: float  # 0-100 tactical understanding metric  
    consistency_score: float  # 0-100 play consistency metric
    improvement_rate: Optional[float]  # Compared to previous evaluation
    total_games_played: int
    evaluation_time_seconds: float


class BenchmarkSuite:
    """Main evaluation system for model strength assessment."""
    
    def __init__(self, games_per_skill_level: int = 20):
        """
        Initialize benchmark suite.
        
        Args:
            games_per_skill_level: Number of games to play against each skill level
                                 More games = higher statistical confidence
        """
        self.games_per_skill_level = games_per_skill_level
        self.evaluation_history = []
    
    def evaluate_model(self, model, model_name: str = "unknown", 
                      training_step: int = 0) -> BenchmarkReport:
        """
        Run complete evaluation suite against all skill levels.
        
        This is the main entry point for model evaluation.
        Returns detailed performance metrics across all dimensions.
        """
        start_time = datetime.now()
        print(f"\\n=== Evaluating {model_name} (Step {training_step}) ===")
        
        # Create opponents
        skill_levels = create_skill_levels(model)
        skill_results = []
        total_games = 0
        
        # Evaluate against each skill level
        for opponent in skill_levels:
            print(f"Playing {self.games_per_skill_level} games vs {opponent.get_skill_level()}...")
            
            level_result = self._evaluate_against_skill_level(model, opponent)
            skill_results.append(level_result)
            total_games += level_result.games_played
            
            print(f"  Win rate: {level_result.win_rate:.1f}% "
                  f"(±{(level_result.confidence_interval[1] - level_result.win_rate):.1f}%)")
            print(f"  Tactical score: {level_result.avg_tactical_score:.3f}")
        
        # Calculate composite metrics
        strength_score = self._calculate_strength_score(skill_results)
        tactical_score = self._calculate_tactical_score(skill_results)
        consistency_score = self._calculate_consistency_score(skill_results)
        
        # Calculate improvement rate if we have previous results
        improvement_rate = self._calculate_improvement_rate(strength_score)
        
        end_time = datetime.now()
        eval_duration = (end_time - start_time).total_seconds()
        
        # Create report
        report = BenchmarkReport(
            model_name=model_name,
            training_step=training_step,
            timestamp=start_time.isoformat(),
            skill_level_results=skill_results,
            overall_strength_score=strength_score,
            tactical_ability_score=tactical_score,
            consistency_score=consistency_score,
            improvement_rate=improvement_rate,
            total_games_played=total_games,
            evaluation_time_seconds=eval_duration
        )
        
        # Store for improvement tracking
        self.evaluation_history.append(report)
        
        print(f"\\n=== Evaluation Complete ===")
        print(f"Overall Strength: {strength_score:.1f}/100")
        print(f"Tactical Ability: {tactical_score:.1f}/100") 
        print(f"Consistency: {consistency_score:.1f}/100")
        if improvement_rate is not None:
            print(f"Improvement Rate: {improvement_rate:+.1f} points")
        print(f"Total games: {total_games}, Duration: {eval_duration:.1f}s")
        
        return report
    
    def _evaluate_against_skill_level(self, model, opponent: OpponentPlayer) -> SkillLevelResult:
        """Evaluate model against a specific skill level with statistical analysis."""
        game_results = []
        
        # Play games (alternating who goes first for fairness)
        for i in range(self.games_per_skill_level):
            model_plays_first = (i % 2 == 0)
            game_result = play_evaluation_game(model, opponent, model_plays_first)
            game_results.append(game_result)
        
        # Calculate statistics
        wins = sum(1 for result in game_results if result.winner == 1)
        win_rate = (wins / len(game_results)) * 100
        
        avg_game_length = np.mean([r.game_length for r in game_results])
        avg_tactical_score = np.mean([r.model_tactical_score for r in game_results])
        score_std = np.std([r.model_tactical_score for r in game_results])
        
        # Calculate 95% confidence interval for win rate
        n = len(game_results)
        p = win_rate / 100
        margin_of_error = 1.96 * np.sqrt(p * (1 - p) / n) * 100
        ci_lower = max(0, win_rate - margin_of_error)
        ci_upper = min(100, win_rate + margin_of_error)
        
        return SkillLevelResult(
            skill_level=opponent.get_skill_level(),
            games_played=len(game_results),
            win_rate=win_rate,
            avg_game_length=avg_game_length,
            avg_tactical_score=avg_tactical_score,
            score_std=score_std,
            confidence_interval=(ci_lower, ci_upper)
        )
    
    def _calculate_strength_score(self, skill_results: List[SkillLevelResult]) -> float:
        """
        Calculate overall strength score (0-100) based on win rates across skill levels.
        
        Higher skill levels are weighted more heavily.
        Perfect score requires high win rates against strong opponents.
        """
        weights = {
            'random': 1.0,
            'greedy': 2.0,
            'tactical': 4.0,
            'mcts_50': 8.0,
            'mcts_200': 16.0,
            'mcts_500': 32.0,
            'mcts_800': 64.0
        }
        
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for result in skill_results:
            weight = weights.get(result.skill_level, 1.0)
            total_weighted_score += result.win_rate * weight
            total_weight += weight
        
        return total_weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_tactical_score(self, skill_results: List[SkillLevelResult]) -> float:
        """Calculate tactical ability score based on move quality across games."""
        tactical_scores = [result.avg_tactical_score for result in skill_results]
        return np.mean(tactical_scores) * 100 if tactical_scores else 0.0
    
    def _calculate_consistency_score(self, skill_results: List[SkillLevelResult]) -> float:
        """Calculate consistency score (lower standard deviation = higher consistency)."""
        std_scores = [result.score_std for result in skill_results]
        avg_std = np.mean(std_scores) if std_scores else 1.0
        
        # Convert to 0-100 scale where lower std = higher score
        # Assume tactical scores have std between 0 and 0.5
        consistency = max(0, 100 * (1 - 2 * avg_std))
        return min(100, consistency)
    
    def _calculate_improvement_rate(self, current_strength: float) -> Optional[float]:
        """Calculate improvement rate compared to previous evaluation."""
        if len(self.evaluation_history) == 0:
            return None
        
        previous_strength = self.evaluation_history[-1].overall_strength_score
        return current_strength - previous_strength
    
    def save_report(self, report: BenchmarkReport, filename: str):
        """Save evaluation report to JSON file."""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Convert dataclass to dict for JSON serialization
        report_dict = asdict(report)
        
        with open(filename, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        print(f"Evaluation report saved to: {filename}")
    
    def load_report(self, filename: str) -> BenchmarkReport:
        """Load evaluation report from JSON file."""
        with open(filename, 'r') as f:
            report_dict = json.load(f)
        
        # Convert dict back to dataclass
        skill_results = [SkillLevelResult(**sr) for sr in report_dict['skill_level_results']]
        report_dict['skill_level_results'] = skill_results
        
        return BenchmarkReport(**report_dict)
    
    def generate_progress_summary(self) -> str:
        """Generate a text summary of learning progress over time."""
        if len(self.evaluation_history) < 2:
            return "Not enough evaluations to show progress."
        
        summary = "=== Learning Progress Summary ===\\n\\n"
        
        for i, report in enumerate(self.evaluation_history):
            summary += f"Step {report.training_step}: "
            summary += f"Strength={report.overall_strength_score:.1f} "
            summary += f"Tactical={report.tactical_ability_score:.1f} "
            summary += f"Consistency={report.consistency_score:.1f}"
            
            if report.improvement_rate is not None:
                summary += f" (Δ{report.improvement_rate:+.1f})"
            
            summary += "\\n"
        
        # Calculate overall trends
        first_report = self.evaluation_history[0]
        latest_report = self.evaluation_history[-1]
        
        strength_improvement = latest_report.overall_strength_score - first_report.overall_strength_score
        tactical_improvement = latest_report.tactical_ability_score - first_report.tactical_ability_score
        
        summary += f"\\nTotal Improvement: "
        summary += f"Strength: {strength_improvement:+.1f} "
        summary += f"Tactical: {tactical_improvement:+.1f}"
        
        return summary


def run_quick_benchmark(model, model_name: str = "test_model") -> BenchmarkReport:
    """Run a quick benchmark with fewer games per level (for development/testing)."""
    suite = BenchmarkSuite(games_per_skill_level=5)
    return suite.evaluate_model(model, model_name)


def run_full_benchmark(model, model_name: str = "production_model", 
                      training_step: int = 0) -> BenchmarkReport:
    """Run a comprehensive benchmark with full statistical significance."""
    suite = BenchmarkSuite(games_per_skill_level=50)
    return suite.evaluate_model(model, model_name, training_step)