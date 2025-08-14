"""Configuration Management System for AlphaPente Training Pipeline

This module provides centralized configuration management for the AlphaPente training pipeline.
It loads parameters from JSON files and provides dynamic model loading capabilities.

Key Features:
- JSON-based configuration with nested parameter access
- Dynamic model class loading from string specifications
- Configuration update and persistence
- Default value handling
- SQL query template generation for different sampling strategies

Usage:
    from config_loader import config
    simulations = config.get('generation', 'simulations', default=100)
    ModelClass = config.get_model_class('resnet')

Configuration Structure:
    {
        "generation": {"simulations": 100, "model_type": "simple"},
        "training": {"batch_size": 32, "sample_strategy": "top_random"},
        "pipeline": {"generation_cycles": 3},
        "models": {"simple": {"class": "GomokuSimpleNN", "file": "models.gomoku_simple_nn"}}
    }
"""

import json
import os
from typing import Dict, Any
import importlib


class ConfigLoader:
    """Load and manage pipeline configuration."""
    
    def __init__(self, config_path: str = "config/pipeline.json"):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return json.load(f)
    
    def get(self, section: str, key: str = None, default=None):
        """Get configuration value."""
        try:
            if key is None:
                return self.config.get(section, default)
            return self.config[section].get(key, default)
        except KeyError:
            return default
    
    def get_model_class(self, model_type: str):
        """Dynamically load model class from config."""
        model_config = self.config["models"].get(model_type)
        if not model_config:
            raise ValueError(f"Unknown model type: {model_type}")
        
        module = importlib.import_module(model_config["file"])
        return getattr(module, model_config["class"])
    
    def get_generation_config(self) -> Dict[str, Any]:
        """Get generation-specific configuration."""
        return self.config.get("generation", {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training-specific configuration."""
        return self.config.get("training", {})
    
    def get_pipeline_config(self) -> Dict[str, Any]:
        """Get pipeline coordination configuration."""
        return self.config.get("pipeline", {})
    
    def update_config(self, section: str, key: str, value: Any):
        """Update configuration value."""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
    
    def save_config(self):
        """Save current configuration back to file."""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def get_sample_strategy_query(self, strategy: str, table_name: str) -> str:
        """Get SQL query for sampling strategy."""
        strategies = self.config["database"]["sample_strategies"]
        query_template = strategies.get(strategy)
        
        if not query_template:
            raise ValueError(f"Unknown sampling strategy: {strategy}")
        
        # Replace placeholders
        return query_template.format(table=table_name)


# Global config instance - automatically loads from default config/pipeline.json
# This can be imported directly: from config_loader import config
config = ConfigLoader()