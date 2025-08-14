"""Performance Tracking and Visualization for Model Evaluation

This module provides tools to track performance over time, generate learning curves,
and create visual reports showing model improvement across training cycles.

Key Features:
- Automated performance tracking during training pipeline
- Learning curve generation with smooth trend lines
- Comparative analysis between different models/configurations
- Export capabilities for external analysis
- Integration with training schedule for continuous monitoring

The tracker maintains a persistent database of evaluation results and can generate
various visualizations to demonstrate learning progress over time.
"""

import numpy as np
import json
import os
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import asdict
import sqlite3

from .benchmark_suite import BenchmarkReport, BenchmarkSuite


class PerformanceTracker:
    """Tracks model performance over time with persistent storage."""
    
    def __init__(self, tracking_db_path: str = "evaluation/performance_history.db"):
        """Initialize performance tracker with persistent database."""
        self.db_path = tracking_db_path
        self._setup_database()
        self.current_session_results = []
    
    def _setup_database(self):
        """Create database tables for performance tracking."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Main evaluation results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                training_step INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                overall_strength_score REAL NOT NULL,
                tactical_ability_score REAL NOT NULL,
                consistency_score REAL NOT NULL,
                improvement_rate REAL,
                total_games_played INTEGER NOT NULL,
                evaluation_time_seconds REAL NOT NULL,
                raw_report TEXT NOT NULL  -- JSON serialized BenchmarkReport
            )
        """)
        
        # Skill level results table for detailed analysis
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS skill_level_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                evaluation_id INTEGER NOT NULL,
                skill_level TEXT NOT NULL,
                games_played INTEGER NOT NULL,
                win_rate REAL NOT NULL,
                avg_game_length REAL NOT NULL,
                avg_tactical_score REAL NOT NULL,
                score_std REAL NOT NULL,
                ci_lower REAL NOT NULL,
                ci_upper REAL NOT NULL,
                FOREIGN KEY (evaluation_id) REFERENCES evaluations (id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def record_evaluation(self, report: BenchmarkReport):
        """Store evaluation results in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Insert main evaluation record
        cursor.execute("""
            INSERT INTO evaluations (
                model_name, training_step, timestamp, overall_strength_score,
                tactical_ability_score, consistency_score, improvement_rate,
                total_games_played, evaluation_time_seconds, raw_report
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            report.model_name,
            report.training_step,
            report.timestamp,
            report.overall_strength_score,
            report.tactical_ability_score,
            report.consistency_score,
            report.improvement_rate,
            report.total_games_played,
            report.evaluation_time_seconds,
            json.dumps(asdict(report))
        ))
        
        evaluation_id = cursor.lastrowid
        
        # Insert skill level results
        for skill_result in report.skill_level_results:
            cursor.execute("""
                INSERT INTO skill_level_results (
                    evaluation_id, skill_level, games_played, win_rate,
                    avg_game_length, avg_tactical_score, score_std,
                    ci_lower, ci_upper
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                evaluation_id,
                skill_result.skill_level,
                skill_result.games_played,
                skill_result.win_rate,
                skill_result.avg_game_length,
                skill_result.avg_tactical_score,
                skill_result.score_std,
                skill_result.confidence_interval[0],
                skill_result.confidence_interval[1]
            ))
        
        conn.commit()
        conn.close()
        
        # Add to current session for immediate analysis
        self.current_session_results.append(report)
        
        print(f"Recorded evaluation for {report.model_name} at step {report.training_step}")
    
    def get_learning_curve_data(self, model_name: str) -> Dict[str, List]:
        """Get learning curve data for a specific model."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT training_step, overall_strength_score, tactical_ability_score, 
                   consistency_score, timestamp
            FROM evaluations
            WHERE model_name = ?
            ORDER BY training_step
        """, (model_name,))
        
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            return {}
        
        return {
            'training_steps': [r[0] for r in results],
            'strength_scores': [r[1] for r in results],
            'tactical_scores': [r[2] for r in results],
            'consistency_scores': [r[3] for r in results],
            'timestamps': [r[4] for r in results]
        }
    
    def get_skill_level_progression(self, model_name: str, skill_level: str) -> Dict[str, List]:
        """Get win rate progression against a specific skill level over time."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT e.training_step, s.win_rate, s.ci_lower, s.ci_upper, s.avg_tactical_score
            FROM evaluations e
            JOIN skill_level_results s ON e.id = s.evaluation_id
            WHERE e.model_name = ? AND s.skill_level = ?
            ORDER BY e.training_step
        """, (model_name, skill_level))
        
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            return {}
        
        return {
            'training_steps': [r[0] for r in results],
            'win_rates': [r[1] for r in results],
            'ci_lower': [r[2] for r in results],
            'ci_upper': [r[3] for r in results],
            'tactical_scores': [r[4] for r in results]
        }
    
    def generate_text_report(self, model_name: str) -> str:
        """Generate a comprehensive text report of model progress."""
        curve_data = self.get_learning_curve_data(model_name)
        
        if not curve_data:
            return f"No evaluation data found for model: {model_name}"
        
        report = f"=== Performance Report for {model_name} ===\\n\\n"
        
        # Overall progress summary
        initial_strength = curve_data['strength_scores'][0]
        latest_strength = curve_data['strength_scores'][-1]
        total_improvement = latest_strength - initial_strength
        
        initial_step = curve_data['training_steps'][0]
        latest_step = curve_data['training_steps'][-1]
        steps_trained = latest_step - initial_step
        
        report += f"Training Progress:\\n"
        report += f"  Steps: {initial_step} → {latest_step} (+{steps_trained})\\n"
        report += f"  Overall Strength: {initial_strength:.1f} → {latest_strength:.1f} ({total_improvement:+.1f})\\n"
        report += f"  Tactical Ability: {curve_data['tactical_scores'][0]:.1f} → {curve_data['tactical_scores'][-1]:.1f}\\n"
        report += f"  Consistency: {curve_data['consistency_scores'][0]:.1f} → {curve_data['consistency_scores'][-1]:.1f}\\n\\n"
        
        # Learning rate analysis
        if len(curve_data['strength_scores']) > 3:
            recent_improvement = curve_data['strength_scores'][-1] - curve_data['strength_scores'][-4]
            report += f"Recent Learning Rate: {recent_improvement:+.1f} points over last 3 evaluations\\n\\n"
        
        # Skill level breakdown
        report += "Performance vs Skill Levels:\\n"
        skill_levels = ['random', 'greedy', 'tactical', 'mcts_50', 'mcts_200', 'mcts_500', 'mcts_800']
        
        for skill in skill_levels:
            skill_data = self.get_skill_level_progression(model_name, skill)
            if skill_data:
                initial_wr = skill_data['win_rates'][0]
                latest_wr = skill_data['win_rates'][-1]
                improvement = latest_wr - initial_wr
                
                report += f"  {skill:12}: {initial_wr:5.1f}% → {latest_wr:5.1f}% ({improvement:+5.1f}%)\\n"
        
        # Performance milestones
        report += "\\nMilestones Achieved:\\n"
        milestones = [
            (50, "Beats random player consistently"),
            (60, "Shows tactical understanding"),
            (70, "Competitive intermediate play"),
            (80, "Strong tactical player"),
            (90, "Expert-level performance")
        ]
        
        for threshold, description in milestones:
            if latest_strength >= threshold:
                report += f"  ✓ {description} (Strength: {latest_strength:.1f})\\n"
            else:
                report += f"  ✗ {description} (Need: {threshold}, Current: {latest_strength:.1f})\\n"
        
        return report
    
    def export_csv_data(self, model_name: str, filename: str):
        """Export evaluation data to CSV for external analysis."""
        conn = sqlite3.connect(self.db_path)
        
        # Export main metrics
        query = """
            SELECT training_step, timestamp, overall_strength_score, 
                   tactical_ability_score, consistency_score, improvement_rate,
                   total_games_played, evaluation_time_seconds
            FROM evaluations
            WHERE model_name = ?
            ORDER BY training_step
        """
        
        import pandas as pd
        df = pd.read_sql_query(query, conn, params=(model_name,))
        
        if not df.empty:
            df.to_csv(filename, index=False)
            print(f"Exported {len(df)} evaluation records to {filename}")
        else:
            print(f"No data found for model: {model_name}")
        
        conn.close()
    
    def get_model_comparison(self, model_names: List[str]) -> str:
        """Generate comparison report between different models."""
        if len(model_names) < 2:
            return "Need at least 2 models for comparison"
        
        report = f"=== Model Comparison Report ===\\n\\n"
        
        comparison_data = {}
        for model_name in model_names:
            curve_data = self.get_learning_curve_data(model_name)
            if curve_data:
                comparison_data[model_name] = {
                    'final_strength': curve_data['strength_scores'][-1],
                    'final_tactical': curve_data['tactical_scores'][-1],
                    'total_steps': curve_data['training_steps'][-1] - curve_data['training_steps'][0],
                    'improvement': curve_data['strength_scores'][-1] - curve_data['strength_scores'][0]
                }
        
        if not comparison_data:
            return "No data available for comparison"
        
        # Sort models by final strength
        sorted_models = sorted(comparison_data.items(), 
                             key=lambda x: x[1]['final_strength'], reverse=True)
        
        report += f"Ranking by Final Strength:\\n"
        for i, (model, data) in enumerate(sorted_models, 1):
            report += f"  {i}. {model:20}: {data['final_strength']:6.1f} "
            report += f"(Tactical: {data['final_tactical']:5.1f}, "
            report += f"Improvement: {data['improvement']:+5.1f})\\n"
        
        report += f"\\nLearning Efficiency (Improvement per 1000 steps):\\n"
        efficiency_ranking = sorted(comparison_data.items(),
                                  key=lambda x: x[1]['improvement'] / max(1, x[1]['total_steps'] / 1000),
                                  reverse=True)
        
        for i, (model, data) in enumerate(efficiency_ranking, 1):
            steps_k = data['total_steps'] / 1000
            efficiency = data['improvement'] / max(0.1, steps_k)
            report += f"  {i}. {model:20}: {efficiency:6.1f} points/1k steps\\n"
        
        return report


def integrate_with_schedule(model, model_name: str, training_step: int, 
                          tracker: PerformanceTracker) -> BenchmarkReport:
    """
    Integration function for the training schedule to automatically run evaluations.
    
    This function should be called from the schedule script after training cycles.
    """
    print(f"\\n🔍 Running model evaluation at training step {training_step}...")
    
    # Run evaluation (adjust games per level based on time constraints)
    suite = BenchmarkSuite(games_per_skill_level=20)  # Balanced: good stats but reasonable time
    report = suite.evaluate_model(model, model_name, training_step)
    
    # Record results
    tracker.record_evaluation(report)
    
    # Generate and display progress summary
    progress_summary = tracker.generate_text_report(model_name)
    print("\\n" + progress_summary)
    
    # Save detailed report
    os.makedirs("evaluation/reports", exist_ok=True)
    report_file = f"evaluation/reports/{model_name}_step_{training_step}.json"
    suite.save_report(report, report_file)
    
    return report