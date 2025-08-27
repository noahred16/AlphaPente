# AlphaPente
Pente AI agent inspired by AlphaZero

```
mcts_board_games/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── games/
│   │   ├── __init__.py
│   │   ├── base_game.py          # Abstract base class
│   │   ├── gomoku.py             
│   │   ├── pente.py              
│   │   └── game_encoder.py       # Convert game states to NN input
```

```
mcts_board_games/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── games/
│   │   ├── __init__.py
│   │   ├── base_game.py          # Abstract base class
│   │   ├── gomoku.py             
│   │   ├── pente.py              
│   │   └── game_encoder.py       # Convert game states to NN input
│   │
│   ├── mcts/
│   │   ├── __init__.py
│   │   ├── node.py               
│   │   ├── base_mcts.py          # Abstract MCTS class
│   │   ├── vanilla_mcts.py       # Traditional MCTS with random rollouts
│   │   ├── neural_mcts.py        # AlphaZero-style MCTS with NN
│   │   └── policies.py           
│   │
│   ├── neural/
│   │   ├── __init__.py
│   │   ├── architectures/
│   │   │   ├── __init__.py
│   │   │   ├── base_network.py   # Abstract NN class
│   │   │   ├── resnet.py         # ResNet architecture
│   │   │   ├── simple_cnn.py     # Simpler CNN for quick testing
│   │   │   └── attention_net.py  # Optional attention-based architecture
│   │   │
│   │   ├── features.py           # Feature extraction for board states
│   │   ├── model_manager.py      # Save/load models, versioning
│   │   └── inference.py          # Batched inference, caching
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── self_play.py          # Self-play game generation
│   │   ├── replay_buffer.py      # Experience replay storage
│   │   ├── trainer.py            # Neural network training loop
│   │   ├── data_augmentation.py  # Board symmetries/rotations
│   │   └── curriculum.py         # Curriculum learning strategies
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── game_record.py        # Store/load game records
│   │   ├── dataset.py            # PyTorch/TF dataset classes
│   │   └── statistics.py         # Track training metrics
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── elo_system.py         # ELO rating for different versions
│   │   ├── match_maker.py        # Organize matches between agents
│   │   ├── metrics.py            # Win rate, move accuracy, etc.
│   │   └── analysis.py           # Game analysis tools
│   │
│   ├── players/
│   │   ├── __init__.py
│   │   ├── base_player.py        
│   │   ├── human_player.py       
│   │   ├── mcts_player.py        # Can use vanilla or neural MCTS
│   │   ├── neural_player.py      # Direct NN policy (no search)
│   │   └── random_player.py      
│   │
│   └── utils/
│       ├── __init__.py
│       ├── board.py              
│       ├── config.py             
│       ├── logger.py             
│       └── device_manager.py     # GPU/CPU device management
│
├── configs/
│   ├── training/
│   │   ├── base_training.yaml    # Training hyperparameters
│   │   ├── self_play.yaml        # Self-play settings
│   │   └── neural_arch.yaml      # Network architecture configs
│   │
│   ├── mcts/
│   │   ├── vanilla_mcts.yaml    
│   │   └── neural_mcts.yaml      # PUCT constants, exploration params
│   │
│   └── games/
│       ├── gomoku.yaml
│       └── pente.yaml
│
├── data/
│   ├── games/                    # Stored self-play games
│   │   ├── gomoku/
│   │   └── pente/
│   ├── models/                   # Saved neural network checkpoints
│   │   ├── gomoku/
│   │   └── pente/
│   └── replay_buffers/           # Training data
│       ├── gomoku/
│       └── pente/
│
├── experiments/
│   ├── train_from_scratch.py    # Full training pipeline
│   ├── compare_versions.py      # Compare different model versions
│   ├── ablation_studies.py      # Test impact of different components
│   └── hyperparameter_search.py # Automated hyperparameter tuning
│
└── scripts/
    ├── collect_self_play.py     # Generate training data
    ├── train_model.py           # Train neural network
    ├── evaluate_model.py        # Test model performance
    └── play_interactive.py      # Play against the AI
```