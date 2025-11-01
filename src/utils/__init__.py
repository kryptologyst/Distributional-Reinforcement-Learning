"""
Utility functions and classes for distributional RL.

This module provides logging, visualization, and other utility functions.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, deque
import json
import os
from datetime import datetime
import torch


class MetricsLogger:
    """
    Logger for tracking training metrics and statistics.
    """
    
    def __init__(self, log_dir: str = "logs"):
        """
        Initialize the metrics logger.
        
        Args:
            log_dir: Directory to save logs
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.metrics = defaultdict(list)
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"training_{timestamp}.json")
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """
        Log a metric value.
        
        Args:
            name: Name of the metric
            value: Value to log
            step: Optional step number
        """
        self.metrics[name].append({
            'value': value,
            'step': step or len(self.metrics[name]),
            'timestamp': datetime.now().isoformat()
        })
    
    def log_episode(self, reward: float, length: int, step: Optional[int] = None) -> None:
        """
        Log episode statistics.
        
        Args:
            reward: Episode reward
            length: Episode length
            step: Optional step number
        """
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        
        self.log_metric("episode_reward", reward, step)
        self.log_metric("episode_length", length, step)
        
        # Log running averages
        if len(self.episode_rewards) >= 10:
            avg_reward = np.mean(list(self.episode_rewards)[-10:])
            avg_length = np.mean(list(self.episode_lengths)[-10:])
            self.log_metric("avg_reward_10", avg_reward, step)
            self.log_metric("avg_length_10", avg_length, step)
    
    def save_logs(self) -> None:
        """Save logs to file."""
        log_data = {
            'metrics': dict(self.metrics),
            'episode_rewards': list(self.episode_rewards),
            'episode_lengths': list(self.episode_lengths),
            'summary': self.get_summary()
        }
        
        with open(self.log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.episode_rewards:
            return {}
        
        rewards = list(self.episode_rewards)
        lengths = list(self.episode_lengths)
        
        return {
            'total_episodes': len(rewards),
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'avg_length': np.mean(lengths),
            'std_length': np.std(lengths),
            'min_length': np.min(lengths),
            'max_length': np.max(lengths)
        }


class Visualizer:
    """
    Visualization utilities for distributional RL.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Default figure size
        """
        self.figsize = figsize
        plt.style.use('seaborn-v0_8')
    
    def plot_learning_curves(self, logger: MetricsLogger, 
                            save_path: Optional[str] = None) -> None:
        """
        Plot learning curves for training metrics.
        
        Args:
            logger: Metrics logger with training data
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # Episode rewards
        if 'episode_reward' in logger.metrics:
            rewards = [m['value'] for m in logger.metrics['episode_reward']]
            axes[0, 0].plot(rewards, alpha=0.6, label='Episode Reward')
            
            if 'avg_reward_10' in logger.metrics:
                avg_rewards = [m['value'] for m in logger.metrics['avg_reward_10']]
                axes[0, 0].plot(avg_rewards, label='10-Episode Average', linewidth=2)
            
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Episode lengths
        if 'episode_length' in logger.metrics:
            lengths = [m['value'] for m in logger.metrics['episode_length']]
            axes[0, 1].plot(lengths, alpha=0.6, label='Episode Length')
            
            if 'avg_length_10' in logger.metrics:
                avg_lengths = [m['value'] for m in logger.metrics['avg_length_10']]
                axes[0, 1].plot(avg_lengths, label='10-Episode Average', linewidth=2)
            
            axes[0, 1].set_title('Episode Lengths')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Length')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Training loss (if available)
        if 'loss' in logger.metrics:
            losses = [m['value'] for m in logger.metrics['loss']]
            axes[1, 0].plot(losses, alpha=0.6)
            axes[1, 0].set_title('Training Loss')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Epsilon decay (if available)
        if 'epsilon' in logger.metrics:
            epsilons = [m['value'] for m in logger.metrics['epsilon']]
            axes[1, 1].plot(epsilons, alpha=0.6)
            axes[1, 1].set_title('Epsilon Decay')
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Epsilon')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_distribution(self, distributions: np.ndarray, 
                         support: np.ndarray, title: str = "Return Distribution",
                         save_path: Optional[str] = None) -> None:
        """
        Plot return distributions.
        
        Args:
            distributions: Probability distributions
            support: Support values for the distributions
            title: Plot title
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(10, 6))
        
        for i, dist in enumerate(distributions):
            plt.plot(support, dist, label=f'Action {i}', linewidth=2)
        
        plt.title(title)
        plt.xlabel('Return Value')
        plt.ylabel('Probability')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_q_values(self, q_values: np.ndarray, 
                     title: str = "Q-Values", 
                     save_path: Optional[str] = None) -> None:
        """
        Plot Q-values heatmap.
        
        Args:
            q_values: Q-values matrix
            title: Plot title
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(8, 6))
        
        if q_values.ndim == 2:
            sns.heatmap(q_values, annot=True, fmt='.2f', cmap='viridis')
        else:
            plt.plot(q_values)
        
        plt.title(title)
        plt.xlabel('Action')
        plt.ylabel('State')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_checkpoint_dir(base_dir: str = "checkpoints") -> str:
    """
    Create a checkpoint directory with timestamp.
    
    Args:
        base_dir: Base directory for checkpoints
        
    Returns:
        Path to the created checkpoint directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir


if __name__ == "__main__":
    # Test the utilities
    print("Testing utilities...")
    
    # Test metrics logger
    logger = MetricsLogger()
    
    for i in range(10):
        reward = np.random.randn() * 10
        length = np.random.randint(50, 200)
        logger.log_episode(reward, length, i)
        logger.log_metric("loss", np.random.exponential(0.1), i)
        logger.log_metric("epsilon", 0.1 * (0.99 ** i), i)
    
    print("Metrics summary:", logger.get_summary())
    
    # Test visualizer
    visualizer = Visualizer()
    visualizer.plot_learning_curves(logger)
    
    print("Utilities test completed!")
