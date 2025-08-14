#!/usr/bin/env python3
"""Standalone Model Evaluation Script

This script provides a command-line interface to evaluate model strength
using the comprehensive benchmark suite. It can be run independently
or integrated into training pipelines.

Usage Examples:
    # Quick evaluation (5 games per skill level)
    python evaluate_model.py --quick --model-type simple

    # Full evaluation (50 games per skill level)  
    python evaluate_model.py --full --model-type resnet --checkpoint ./checkpoints/resnet_model.pt

    # Compare with previous evaluation
    python evaluate_model.py --model-type simple --compare-with evaluation/reports/simple_step_1000.json

    # Export results for analysis
    python evaluate_model.py --model-type simple --export-csv results.csv

Features:
- Multiple evaluation modes (quick/full)
- Model comparison capabilities
- Progress tracking across training cycles
- Detailed statistical reports
- CSV export for external analysis
"""

import argparse
import os
import sys
import torch
import json
from typing import Optional

# Add project root to path
sys.path.append('.')

from config_loader import config
from evaluation import run_quick_benchmark, run_full_benchmark, BenchmarkSuite
from evaluation.performance_tracker import PerformanceTracker


def load_model(model_type: str, checkpoint_path: Optional[str] = None):
    """Load model with optional checkpoint."""
    # Create model
    ModelClass = config.get_model_class(model_type)
    model = ModelClass(7)
    
    # Determine checkpoint path
    if checkpoint_path is None:
        checkpoint_dir = config.get("pipeline", "checkpoint_dir", "./checkpoints")
        checkpoint_path = os.path.join(checkpoint_dir, f"{model_type}_model.pt")
    
    # Load checkpoint if it exists
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        
        # Get training info if available
        epoch_info = ""
        if "epoch" in checkpoint:
            epoch_info = f" (epoch {checkpoint['epoch']})"
        elif "policy_loss" in checkpoint:
            epoch_info = f" (loss: {checkpoint['policy_loss']:.4f})"
        
        print(f"✓ Model loaded successfully{epoch_info}")
    else:
        print(f"⚠ No checkpoint found at {checkpoint_path}")
        print("⚠ Evaluating untrained model with random weights")
    
    return model


def compare_with_previous(current_report, comparison_file: str):
    """Compare current evaluation with a previous report."""
    if not os.path.exists(comparison_file):
        print(f"⚠ Comparison file not found: {comparison_file}")
        return
    
    print(f"\\n=== Comparison with {comparison_file} ===")
    
    # Load previous report
    suite = BenchmarkSuite()
    try:
        previous_report = suite.load_report(comparison_file)
        
        # Compare key metrics
        strength_diff = current_report.overall_strength_score - previous_report.overall_strength_score
        tactical_diff = current_report.tactical_ability_score - previous_report.tactical_ability_score
        consistency_diff = current_report.consistency_score - previous_report.consistency_score
        
        print(f"Overall Strength: {previous_report.overall_strength_score:.1f} → {current_report.overall_strength_score:.1f} ({strength_diff:+.1f})")
        print(f"Tactical Ability: {previous_report.tactical_ability_score:.1f} → {current_report.tactical_ability_score:.1f} ({tactical_diff:+.1f})")
        print(f"Consistency: {previous_report.consistency_score:.1f} → {current_report.consistency_score:.1f} ({consistency_diff:+.1f})")
        
        # Compare skill level performance
        print(f"\\nSkill Level Improvements:")
        prev_skills = {sr.skill_level: sr.win_rate for sr in previous_report.skill_level_results}
        curr_skills = {sr.skill_level: sr.win_rate for sr in current_report.skill_level_results}
        
        for skill_level in prev_skills:
            if skill_level in curr_skills:
                diff = curr_skills[skill_level] - prev_skills[skill_level]
                print(f"  {skill_level:12}: {prev_skills[skill_level]:5.1f}% → {curr_skills[skill_level]:5.1f}% ({diff:+5.1f}%)")
        
    except Exception as e:
        print(f"⚠ Error loading comparison file: {e}")


def main():
    parser = argparse.ArgumentParser(description="AlphaPente Model Evaluation Tool")
    
    # Model configuration
    parser.add_argument("--model-type", type=str, choices=["simple", "resnet"], 
                       default="simple", help="Model type to evaluate")
    parser.add_argument("--checkpoint", type=str, 
                       help="Path to model checkpoint (default: auto-detect)")
    
    # Evaluation mode
    eval_group = parser.add_mutually_exclusive_group()
    eval_group.add_argument("--quick", action="store_true",
                           help="Quick evaluation (5 games per skill level)")
    eval_group.add_argument("--full", action="store_true", 
                           help="Full evaluation (50 games per skill level)")
    parser.add_argument("--games-per-level", type=int, default=20,
                       help="Custom number of games per skill level")
    
    # Analysis options
    parser.add_argument("--training-step", type=int, default=0,
                       help="Training step number for tracking")
    parser.add_argument("--compare-with", type=str,
                       help="Path to previous evaluation report for comparison")
    parser.add_argument("--export-csv", type=str,
                       help="Export evaluation history to CSV file")
    
    # Output options
    parser.add_argument("--save-report", type=str,
                       help="Save detailed report to JSON file")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress detailed output")
    
    # Performance tracking
    parser.add_argument("--track-performance", action="store_true",
                       help="Save results to performance tracking database")

    args = parser.parse_args()

    # Load model
    try:
        model = load_model(args.model_type, args.checkpoint)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return 1

    # Determine evaluation mode
    if args.quick:
        games_per_level = 5
        mode_name = "quick"
    elif args.full:
        games_per_level = 50
        mode_name = "full"
    else:
        games_per_level = args.games_per_level
        mode_name = "custom"

    if not args.quiet:
        print(f"\\n🎯 Running {mode_name} evaluation ({games_per_level} games per skill level)...")

    # Run evaluation
    try:
        suite = BenchmarkSuite(games_per_skill_level=games_per_level)
        report = suite.evaluate_model(
            model, 
            model_name=f"{args.model_type}_evaluation",
            training_step=args.training_step
        )
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return 1

    # Save report if requested
    if args.save_report:
        suite.save_report(report, args.save_report)

    # Performance tracking
    if args.track_performance:
        tracker = PerformanceTracker()
        tracker.record_evaluation(report)
        
        # Show progress summary
        progress_summary = tracker.generate_text_report(f"{args.model_type}_evaluation")
        print("\\n" + progress_summary)

    # Comparison with previous evaluation
    if args.compare_with:
        compare_with_previous(report, args.compare_with)

    # Export CSV data
    if args.export_csv:
        tracker = PerformanceTracker()
        tracker.export_csv_data(f"{args.model_type}_evaluation", args.export_csv)

    # Final summary
    if not args.quiet:
        print(f"\\n✅ Evaluation completed successfully!")
        print(f"📊 Model Strength: {report.overall_strength_score:.1f}/100")
        print(f"🎯 Tactical Ability: {report.tactical_ability_score:.1f}/100")
        print(f"📈 Consistency: {report.consistency_score:.1f}/100")
        print(f"⏱️  Total Time: {report.evaluation_time_seconds:.1f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())