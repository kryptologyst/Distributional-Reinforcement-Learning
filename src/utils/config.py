"""
Configuration management for distributional RL.

This module handles loading and managing configuration files.
"""

from typing import Dict, Any, Optional
import yaml
import os
from pathlib import Path


class Config:
    """
    Configuration manager for the distributional RL project.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()
    
    def _get_default_config_path(self) -> str:
        """Get the default configuration file path."""
        current_dir = Path(__file__).parent.parent
        return str(current_dir / "config" / "default.yaml")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'agent.lr')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'agent.lr')
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Save configuration to file.
        
        Args:
            path: Optional path to save to
        """
        save_path = path or self.config_path
        
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of updates
        """
        self._update_dict(self.config, updates)
    
    def _update_dict(self, base_dict: Dict[str, Any], 
                    updates: Dict[str, Any]) -> None:
        """Recursively update dictionary."""
        for key, value in updates.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                self._update_dict(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dictionary-style assignment."""
        self.set(key, value)
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"Config({self.config_path})"


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration object
    """
    return Config(config_path)


if __name__ == "__main__":
    # Test configuration loading
    print("Testing configuration...")
    
    try:
        config = load_config()
        print(f"Loaded config from: {config.config_path}")
        print(f"Environment: {config.get('env.name')}")
        print(f"Agent algorithm: {config.get('agent.algorithm')}")
        print(f"Learning rate: {config.get('agent.lr')}")
        print(f"Episodes: {config.get('training.episodes')}")
        
        # Test setting values
        config.set('agent.lr', 0.0005)
        print(f"Updated learning rate: {config.get('agent.lr')}")
        
    except FileNotFoundError as e:
        print(f"Configuration file not found: {e}")
        print("Please ensure config/default.yaml exists")
    
    print("Configuration test completed!")
