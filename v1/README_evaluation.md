# AlphaPente Model Evaluation System

A comprehensive evaluation framework that provides **smooth, continuous metrics** for tracking model learning progress over time, replacing binary pass/fail tests with detailed performance analysis.

## 🎯 **Key Improvements Over Binary Tests**

### **Before: Binary Tests**
- ✗ Pass/fail tactical patterns
- ✗ No learning progression visibility  
- ✗ Hard to see gradual improvement
- ✗ Limited statistical confidence

### **After: Smooth Evaluation Curves**
- ✅ **Win rates** against 7 skill levels (0-100%)
- ✅ **Tactical scoring** continuous 0-1 metrics
- ✅ **Statistical confidence** with error bars
- ✅ **Learning curves** showing clear progress over time
- ✅ **Automated tracking** integrated with training pipeline

## 📊 **Evaluation Metrics**

### **1. Overall Strength Score (0-100)**
Composite metric weighing performance against different skill levels:
- **50+**: Beats random/greedy players consistently
- **60+**: Shows tactical understanding
- **70+**: Competitive intermediate play  
- **80+**: Strong tactical player
- **90+**: Expert-level performance

### **2. Skill Level Progression**
Win rates against increasingly difficult opponents:
1. **Random Player** - Completely random moves
2. **Greedy Player** - Center-focused heuristic
3. **Tactical Player** - Blocks threats, creates attacks
4. **MCTS-50** - 50 simulation MCTS
5. **MCTS-200** - 200 simulation MCTS  
6. **MCTS-500** - 500 simulation MCTS
7. **MCTS-800** - 800 simulation MCTS

### **3. Tactical Ability Score (0-100)**
Continuous measure of move quality:
- Center play bonus (early game)
- Threat creation detection
- Threat blocking accuracy
- Strategic positioning

### **4. Consistency Score (0-100)**  
Measures play reliability (lower standard deviation = higher consistency)

## 🚀 **Quick Start**

### **Standalone Evaluation**
```bash
# Quick test (5 games per level, ~2 minutes)
python evaluate_model.py --quick --model-type simple

# Full evaluation (50 games per level, ~20 minutes)
python evaluate_model.py --full --model-type resnet

# Custom evaluation with specific checkpoint
python evaluate_model.py --games-per-level 20 --checkpoint ./my_model.pt
```

### **Integrated Pipeline Evaluation**
The training pipeline automatically runs evaluations after each training cycle:
```bash
# Pipeline includes automatic evaluation
./schedule_v2.sh
```

### **Progress Tracking**
```bash
# Compare with previous evaluation
python evaluate_model.py --compare-with evaluation/reports/simple_step_1000.json

# Export learning curves for analysis
python evaluate_model.py --export-csv learning_data.csv --track-performance
```

## 📈 **Sample Learning Progression**

```
=== Learning Progress Summary ===

Step 0:    Strength=15.2 Tactical=23.4 Consistency=45.1
Step 100:  Strength=28.7 Tactical=31.2 Consistency=52.3 (Δ+13.5)
Step 200:  Strength=42.1 Tactical=38.9 Consistency=58.7 (Δ+13.4) 
Step 300:  Strength=56.4 Tactical=47.2 Consistency=64.2 (Δ+14.3)
Step 400:  Strength=68.9 Tactical=55.8 Consistency=71.5 (Δ+12.5)

Total Improvement: Strength: +53.7 Tactical: +32.4

Performance vs Skill Levels:
  random      : 45.0% → 95.0% (+50.0%)
  greedy      : 25.0% → 85.0% (+60.0%)  
  tactical    : 15.0% → 75.0% (+60.0%)
  mcts_50     : 10.0% → 65.0% (+55.0%)
  mcts_200    :  5.0% → 45.0% (+40.0%)
  mcts_500    :  2.0% → 25.0% (+23.0%)
  mcts_800    :  1.0% → 15.0% (+14.0%)

Milestones Achieved:
  ✓ Beats random player consistently (Strength: 68.9)
  ✓ Shows tactical understanding (Strength: 68.9)
  ✗ Competitive intermediate play (Need: 70, Current: 68.9)
```

## 🔧 **Configuration**

### **Evaluation Settings in config/pipeline.json**
```json
{
  "evaluation": {
    "games_per_skill_level": 20,
    "skill_levels": ["random", "greedy", "tactical", "mcts_50", "mcts_200", "mcts_500", "mcts_800"],
    "statistical_confidence": 0.95,
    "track_performance": true
  }
}
```

### **Integration with Training Pipeline**
The evaluation system automatically:
1. **Runs after each training cycle**
2. **Tracks progress in SQLite database**
3. **Generates learning curve data**
4. **Shows statistical confidence intervals**
5. **Compares with previous evaluations**

## 📊 **Output Files**

### **Generated Reports**
- `evaluation/reports/model_step_N.json` - Detailed evaluation reports
- `evaluation/performance_history.db` - SQLite tracking database
- `logs/evaluation_N_M_TIMESTAMP.log` - Evaluation run logs

### **Data Export**  
```bash
# Export learning curves to CSV
python evaluate_model.py --export-csv data.csv --track-performance

# Generate progress report
python -c "
from evaluation.performance_tracker import PerformanceTracker
tracker = PerformanceTracker()
print(tracker.generate_text_report('simple_pipeline'))
"
```

## 🎯 **Statistical Significance**

### **Confidence Intervals**
All win rates include 95% confidence intervals:
```
vs tactical: 65.0% (±4.2%)  # Strong confidence
vs mcts_200: 45.0% (±7.1%)  # Moderate confidence  
vs mcts_800: 15.0% (±5.1%)  # Still learning
```

### **Minimum Games for Significance**
- **Quick evaluation**: 5 games per level (development/testing)
- **Standard evaluation**: 20 games per level (pipeline default) 
- **Full evaluation**: 50 games per level (publication quality)

## 🔍 **Interpreting Results**

### **Healthy Learning Indicators**
- **Smooth upward curves** in strength scores
- **Increasing win rates** against harder opponents
- **Improving tactical scores** over time
- **Stable or improving consistency**

### **Concerning Patterns**
- **Plateaus** in learning curves (may need architecture changes)
- **Decreasing consistency** (possible overfitting)
- **No improvement** against stronger opponents (may need more training data)

## 🔄 **Integration with Training**

The evaluation system seamlessly integrates with the v2 training pipeline:

1. **After each training cycle**, evaluation runs automatically
2. **Progress is tracked** and compared with previous cycles  
3. **Learning curves** are updated in real-time
4. **Statistical reports** show confidence in improvements
5. **Early stopping** possible if performance degrades

This provides **continuous visibility** into model learning with **statistical confidence**, replacing binary tests with smooth, interpretable metrics that clearly show learning progress over time.

## 🎪 **Example Usage in Pipeline**

```bash
# Run full pipeline with integrated evaluation
./schedule_v2.sh

# Check progress at any time
python evaluate_model.py --model-type simple --track-performance

# Compare models
python -c "
from evaluation.performance_tracker import PerformanceTracker
tracker = PerformanceTracker()
print(tracker.get_model_comparison(['simple_pipeline', 'resnet_pipeline']))
"
```

The evaluation system transforms model development from **"did it pass the test?"** to **"how much has it improved and with what confidence?"** - providing the smooth learning curves you need to track genuine progress over time.