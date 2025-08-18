# AlphaPente Training Pipeline v2

Improved asynchronous training pipeline with configuration management and intelligent data sampling.

## Overview

The v2 pipeline consists of:
- **Configurable generation** (`generate_v2.py`) - Creates game data with flexible MCTS parameters
- **Smart training** (`train_v2.py`) - Uses intelligent sampling strategies and model selection
- **Coordinated scheduling** (`schedule_v2.sh`) - Manages cycles and progress tracking
- **JSON configuration** (`config/pipeline.json`) - Centralized parameter management

## Key Improvements

### 1. **Configuration Management**
- JSON-based configuration with parameter inheritance
- Command-line overrides for flexibility
- Support for multiple model types (simple, ResNet)
- Centralized parameter management

### 2. **Intelligent Data Sampling**
Train script now supports 3 sampling strategies:
- `recent`: Most recent records (original behavior)
- `random`: Completely random sampling
- `top_random`: **70% recent + 30% random older records** (recommended)

### 3. **Enhanced Generation**
- Configurable MCTS parameters (simulations, randomness, temperature)
- Progress tracking and checkpointing
- Model type selection
- Detailed logging and statistics

### 4. **Pipeline Coordination**
- Automated generation → training cycles
- Progress monitoring and target checking
- Comprehensive logging with timestamps
- Error handling and cleanup

## Quick Start

### 1. **Basic Pipeline Run**
```bash
./schedule_v2.sh
```

### 2. **Custom Configuration**
```bash
./schedule_v2.sh custom_config.json
```

### 3. **Manual Generation**
```bash
python generate_v2.py --simulations 200 --record-limit 2000 --model-type resnet
```

### 4. **Manual Training** 
```bash
python train_v2.py --epochs 100 --sample-strategy top_random --model-type resnet
```

## Configuration

Edit `config/pipeline.json` for default settings:

```json
{
  "generation": {
    "simulations": 100,
    "record_limit": 1000, 
    "mcts_random": 0.3,
    "model_type": "simple"
  },
  "training": {
    "batch_size": 32,
    "num_epochs": 50,
    "sample_strategy": "top_random",
    "model_type": "simple"
  },
  "pipeline": {
    "generation_cycles": 3,
    "training_cycles_per_generation": 2
  }
}
```

## Model Types

### Simple Model (`simple`)
- Original 3-layer CNN architecture
- Fast training and inference
- Good baseline performance

### ResNet Model (`resnet`) 
- ResNet-style with residual blocks
- Better representation power
- Improved gradient flow

## Sampling Strategies

### `recent` (Original)
- Uses most recent records
- Fast but may overfit to recent games
- Good for initial training

### `random`
- Completely random sampling
- Better diversity but may include low-quality early games
- Good for preventing overfitting

### `top_random` (Recommended)
- **70% from recent records + 30% random from older**
- Balances quality and diversity
- Prevents overfitting while maintaining game quality

## Pipeline Flow

```
1. Generation Cycle 1
   ├── Generate 1000 records with current model
   ├── Training Cycle 1.1 (50 epochs, top_random sampling)
   └── Training Cycle 1.2 (50 epochs, top_random sampling)

2. Generation Cycle 2  
   ├── Generate 1000 more records (2000 total)
   ├── Training Cycle 2.1
   └── Training Cycle 2.2

3. Generation Cycle 3
   └── Continue until target reached...
```

## Logging

All runs generate detailed logs in `logs/`:
- `pipeline_TIMESTAMP.log` - Main coordination log
- `generation_N_TIMESTAMP.log` - Generation cycle details  
- `training_N_M_TIMESTAMP.log` - Training cycle details

## Monitoring Progress

The pipeline tracks:
- Total database records vs target
- Games generated per cycle
- Training loss progression
- Time and performance metrics

## Advanced Usage

### Custom Model Training
```bash
# Train ResNet model with aggressive sampling
python train_v2.py \
    --model-type resnet \
    --sample-strategy random \
    --sample-size 20000 \
    --epochs 100 \
    --batch-size 64
```

### High-Performance Generation
```bash
# Generate with high MCTS simulations
python generate_v2.py \
    --simulations 500 \
    --record-limit 5000 \
    --mcts-random 0.1 \
    --model-type resnet
```

### Model Comparison
```bash
# Train both models in parallel (different terminals)
python train_v2.py --model-type simple --epochs 100 &
python train_v2.py --model-type resnet --epochs 100 &
```

## File Structure

```
AlphaPente/
├── config/
│   └── pipeline.json          # Configuration file
├── generate_v2.py             # Enhanced generation script  
├── train_v2.py               # Enhanced training script
├── schedule_v2.sh            # Pipeline coordinator
├── config_loader.py          # Configuration management
├── logs/                     # Generated log files
└── checkpoints/              # Model checkpoints
    ├── simple_model.pt
    └── resnet_model.pt
```

The v2 pipeline provides much better control, monitoring, and performance compared to the original hardcoded scripts.