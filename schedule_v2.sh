#!/bin/bash

# AlphaPente Training Pipeline Scheduler v2
# Coordinates data generation and model training cycles
#
# This script orchestrates the complete training pipeline by running alternating cycles of:
# 1. Data generation (self-play games using current best model)
# 2. Model training (multiple training cycles on accumulated data)
#
# Key Features:
# - JSON configuration parsing and parameter management
# - Progress tracking with target record monitoring
# - Comprehensive logging with timestamps and separation by cycle
# - Error handling and automatic cleanup
# - Configurable generation and training cycles
# - Database size monitoring and early stopping
#
# Pipeline Flow:
# Generation Cycle 1 → Training Cycle 1.1 → Training Cycle 1.2 →
# Generation Cycle 2 → Training Cycle 2.1 → Training Cycle 2.2 → ...
#
# Configuration:
# Uses config/pipeline.json for default parameters, can be overridden via command line
#
# Logging:
# - logs/pipeline_TIMESTAMP.log: Main coordination log
# - logs/generation_N_TIMESTAMP.log: Individual generation logs  
# - logs/training_N_M_TIMESTAMP.log: Individual training logs
#
# Usage:
#   ./schedule_v2.sh                    # Use default config/pipeline.json
#   ./schedule_v2.sh custom_config.json # Use custom configuration
#
# Example Configuration:
#   generation_cycles: 3                # Number of generation phases
#   training_cycles_per_generation: 2   # Training runs after each generation
#   total_records_target: 50000         # Stop early if target reached
#
# The script monitors database growth and can stop early if targets are met,
# making it suitable for both short experiments and long training runs.

set -e  # Exit on any error

# Configuration
CONFIG_FILE="${1:-config/pipeline.json}"
PYTHON_CMD="python3"
LOG_DIR="logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create log directory
mkdir -p "$LOG_DIR"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_DIR/pipeline_$TIMESTAMP.log"
}

# Error handling
error_exit() {
    log "ERROR: $1"
    exit 1
}

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    error_exit "Configuration file not found: $CONFIG_FILE"
fi

log "Starting AlphaPente training pipeline with config: $CONFIG_FILE"

# Parse configuration (basic JSON parsing with python)
get_config_value() {
    $PYTHON_CMD -c "
import json
with open('$CONFIG_FILE', 'r') as f:
    config = json.load(f)
    
def get_nested(data, path):
    keys = path.split('.')
    for key in keys:
        data = data.get(key, {})
    return data

print(get_nested(config, '$1'))
" 2>/dev/null || echo ""
}

# Get pipeline configuration
GENERATION_CYCLES=$(get_config_value "pipeline.generation_cycles")
TRAINING_CYCLES_PER_GEN=$(get_config_value "pipeline.training_cycles_per_generation")
TOTAL_RECORDS_TARGET=$(get_config_value "pipeline.total_records_target")

# Get model and generation settings
MODEL_TYPE=$(get_config_value "generation.model_type")
SIMULATIONS=$(get_config_value "generation.simulations")
RECORD_LIMIT=$(get_config_value "generation.record_limit")

# Get training settings
TRAIN_MODEL_TYPE=$(get_config_value "training.model_type")
BATCH_SIZE=$(get_config_value "training.batch_size")
EPOCHS=$(get_config_value "training.num_epochs")
SAMPLE_STRATEGY=$(get_config_value "training.sample_strategy")

log "Pipeline Configuration:"
log "  Generation cycles: $GENERATION_CYCLES"
log "  Training cycles per generation: $TRAINING_CYCLES_PER_GEN"
log "  Model type: $MODEL_TYPE"
log "  Simulations per move: $SIMULATIONS"
log "  Records per generation cycle: $RECORD_LIMIT"
log "  Training epochs per cycle: $EPOCHS"
log "  Sample strategy: $SAMPLE_STRATEGY"

# Function to run generation
run_generation() {
    local cycle=$1
    log "=== GENERATION CYCLE $cycle ==="
    
    local gen_log="$LOG_DIR/generation_${cycle}_$TIMESTAMP.log"
    
    log "Starting data generation (targeting $RECORD_LIMIT records)..."
    
    if $PYTHON_CMD generate_v2.py \
        --config "$CONFIG_FILE" \
        --model-type "$MODEL_TYPE" \
        --simulations "$SIMULATIONS" \
        --record-limit "$RECORD_LIMIT" \
        > "$gen_log" 2>&1; then
        
        log "Generation cycle $cycle completed successfully"
        
        # Show summary from generation log
        if grep -q "Generation Complete" "$gen_log"; then
            grep -A 6 "Generation Complete" "$gen_log" | while read line; do
                log "  $line"
            done
        fi
    else
        error_exit "Generation cycle $cycle failed. Check log: $gen_log"
    fi
}

# Function to run training
run_training() {
    local gen_cycle=$1
    local train_cycle=$2
    log "=== TRAINING CYCLE $gen_cycle.$train_cycle ==="
    
    local train_log="$LOG_DIR/training_${gen_cycle}_${train_cycle}_$TIMESTAMP.log"
    
    log "Starting model training ($EPOCHS epochs, $SAMPLE_STRATEGY strategy)..."
    
    if $PYTHON_CMD train_v2.py \
        --config "$CONFIG_FILE" \
        --model-type "$TRAIN_MODEL_TYPE" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --sample-strategy "$SAMPLE_STRATEGY" \
        > "$train_log" 2>&1; then
        
        log "Training cycle $gen_cycle.$train_cycle completed successfully"
        
        # Show training summary
        if grep -q "Training completed" "$train_log"; then
            tail -10 "$train_log" | grep -E "(Avg Policy Loss|Avg Value Loss|Training completed)" | while read line; do
                log "  $line"
            done
        fi
    else
        error_exit "Training cycle $gen_cycle.$train_cycle failed. Check log: $train_log"
    fi
}

# Function to run model evaluation
run_evaluation() {
    local gen_cycle=$1
    local train_cycle=$2
    log "=== MODEL EVALUATION $gen_cycle.$train_cycle ==="
    
    local eval_log="$LOG_DIR/evaluation_${gen_cycle}_${train_cycle}_$TIMESTAMP.log"
    
    # Calculate training step for tracking
    local training_step=$(($gen_cycle * 1000 + $train_cycle * 100))
    
    log "Running model evaluation at training step $training_step..."
    
    if $PYTHON_CMD -c "
import sys
sys.path.append('.')
from evaluation import integrate_with_schedule
from evaluation.performance_tracker import PerformanceTracker
from config_loader import config

# Load model
model_type = config.get('training', 'model_type', 'simple')
ModelClass = config.get_model_class(model_type)
model = ModelClass(7)

# Load checkpoint
import os
import torch
checkpoint_dir = config.get('pipeline', 'checkpoint_dir', './checkpoints')
checkpoint_path = os.path.join(checkpoint_dir, f'{model_type}_model.pt')
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    print(f'Loaded model from {checkpoint_path}')
else:
    print(f'No checkpoint found, evaluating untrained model')

# Run evaluation
tracker = PerformanceTracker()
report = integrate_with_schedule(
    model, 
    model_name='${TRAIN_MODEL_TYPE}_pipeline', 
    training_step=${training_step},
    tracker=tracker
)

print(f'Evaluation complete: Strength={report.overall_strength_score:.1f}/100')
" > "$eval_log" 2>&1; then
        
        log "Model evaluation completed successfully"
        
        # Show evaluation summary
        if grep -q "Evaluation complete" "$eval_log"; then
            grep "Evaluation complete" "$eval_log" | while read line; do
                log "  $line"
            done
        fi
        
        # Show key metrics from evaluation
        if grep -q "Overall Strength:" "$eval_log"; then
            grep -A 3 "Overall Strength:" "$eval_log" | while read line; do
                log "  $line"
            done
        fi
    else
        log "WARNING: Model evaluation failed. Check log: $eval_log"
        log "Continuing with training pipeline..."
    fi
}

# Function to check database size
check_database_progress() {
    local current_records=$($PYTHON_CMD -c "
from db import Database
from settings_loader import DB_PATH, TABLE_NAME, BOARD_SIZE
db = Database(DB_PATH, BOARD_SIZE)
print(db.get_total(TABLE_NAME))
" 2>/dev/null || echo "0")
    
    log "Database currently contains: $current_records records"
    
    if [ "$TOTAL_RECORDS_TARGET" ] && [ "$current_records" -ge "$TOTAL_RECORDS_TARGET" ]; then
        log "Target record count ($TOTAL_RECORDS_TARGET) reached!"
        return 0
    fi
    return 1
}

# Function to cleanup old logs (keep last 10)
cleanup_logs() {
    log "Cleaning up old log files..."
    find "$LOG_DIR" -name "*.log" -type f | sort | head -n -10 | xargs rm -f 2>/dev/null || true
}

# Main pipeline execution
main_pipeline() {
    log "Starting main training pipeline..."
    
    # Initial database check
    check_database_progress
    
    # Main generation and training loop
    for (( gen_cycle=1; gen_cycle<=GENERATION_CYCLES; gen_cycle++ )); do
        
        # Data generation phase
        run_generation $gen_cycle
        
        # Multiple training cycles per generation
        for (( train_cycle=1; train_cycle<=TRAINING_CYCLES_PER_GEN; train_cycle++ )); do
            run_training $gen_cycle $train_cycle
            
            # Run evaluation after each training cycle
            run_evaluation $gen_cycle $train_cycle
        done
        
        # Check if we've reached target
        if check_database_progress; then
            log "Target reached, stopping pipeline early"
            break
        fi
        
        # Add pause between major cycles
        if [ $gen_cycle -lt $GENERATION_CYCLES ]; then
            log "Pausing 30 seconds between cycles..."
            sleep 30
        fi
    done
}

# Trap for cleanup on exit
trap cleanup_logs EXIT

# Check for required scripts
if [ ! -f "generate_v2.py" ]; then
    error_exit "generate_v2.py not found"
fi

if [ ! -f "train_v2.py" ]; then
    error_exit "train_v2.py not found"
fi

# Run the main pipeline
main_pipeline

log "=== PIPELINE COMPLETED ==="
log "Check individual logs in $LOG_DIR/ for detailed output"

final_records=$($PYTHON_CMD -c "
from db import Database
from settings_loader import DB_PATH, TABLE_NAME, BOARD_SIZE
db = Database(DB_PATH, BOARD_SIZE)
print(db.get_total(TABLE_NAME))
" 2>/dev/null || echo "unknown")

log "Final database size: $final_records records"
log "Pipeline logs saved to: $LOG_DIR/pipeline_$TIMESTAMP.log"