#!/usr/bin/env python3
"""
Command-line interface for distributional RL project.

This script provides easy access to training, evaluation, and visualization tools.
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from train import main as train_main
from utils.config import load_config
from utils import Visualizer
import json


def train_command(args):
    """Handle training command."""
    print("Starting training...")
    
    # Prepare arguments for train.py
    sys.argv = ['train.py']
    
    if args.config:
        sys.argv.extend(['--config', args.config])
    if args.env:
        sys.argv.extend(['--env', args.env])
    if args.episodes:
        sys.argv.extend(['--episodes', str(args.episodes)])
    if args.seed:
        sys.argv.extend(['--seed', str(args.seed)])
    if args.checkpoint_dir:
        sys.argv.extend(['--checkpoint-dir', args.checkpoint_dir])
    
    train_main()


def evaluate_command(args):
    """Handle evaluation command."""
    print(f"Evaluating model from {args.model_path}...")
    
    # Load configuration
    config = load_config(args.config)
    
    # Create environment
    from envs import make_env
    env_name = config.get('env.name', 'MiniGrid')
    env = make_env(env_name)
    
    # Create agent
    from agents import C51Agent
    agent = C51Agent(
        state_dim=config.get('agent.state_dim', 2),
        action_dim=config.get('agent.action_dim', 4)
    )
    
    # Load model
    agent.load(args.model_path)
    
    # Evaluate
    from train import evaluate_agent
    avg_reward = evaluate_agent(agent, env, num_episodes=args.episodes)
    
    print(f"Average reward over {args.episodes} episodes: {avg_reward:.2f}")


def visualize_command(args):
    """Handle visualization command."""
    print(f"Visualizing logs from {args.log_path}...")
    
    # Load log data
    with open(args.log_path, 'r') as f:
        log_data = json.load(f)
    
    # Create visualizer
    visualizer = Visualizer()
    
    # Create mock logger with loaded data
    from utils import MetricsLogger
    logger = MetricsLogger()
    logger.metrics = log_data['metrics']
    logger.episode_rewards = log_data['episode_rewards']
    logger.episode_lengths = log_data['episode_lengths']
    
    # Plot learning curves
    visualizer.plot_learning_curves(logger, save_path=args.save_path)


def demo_command(args):
    """Handle demo command."""
    print("Running demo...")
    
    # Import required modules
    from envs import make_env
    from agents import C51Agent
    from utils import set_seed
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Create environment
    env_name = args.env or 'MiniGrid'
    env = make_env(env_name)
    print(f"Created environment: {env_name}")
    
    # Create agent
    state_dim = 2 if env_name == 'MiniGrid' else 4
    action_dim = env.action_space.n
    agent = C51Agent(state_dim=state_dim, action_dim=action_dim)
    
    print(f"Created C51 agent with state_dim={state_dim}, action_dim={action_dim}")
    
    # Run a few episodes
    for episode in range(args.episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        print(f"\nEpisode {episode + 1}:")
        
        for step in range(20):  # Limit steps for demo
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            episode_reward += reward
            print(f"  Step {step + 1}: action={action}, reward={reward:.2f}")
            
            if terminated or truncated:
                break
            
            state = next_state
        
        print(f"  Episode reward: {episode_reward:.2f}")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Distributional Reinforcement Learning CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s train --env MiniGrid --episodes 500
  %(prog)s train --env CartPole-v1 --config config/custom.yaml
  %(prog)s evaluate --model-path checkpoints/model.pt --episodes 10
  %(prog)s visualize --log-path logs/training.json
  %(prog)s demo --env MiniGrid --episodes 3
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a distributional RL agent')
    train_parser.add_argument('--config', type=str, help='Configuration file path')
    train_parser.add_argument('--env', type=str, help='Environment name')
    train_parser.add_argument('--episodes', type=int, help='Number of training episodes')
    train_parser.add_argument('--seed', type=int, help='Random seed')
    train_parser.add_argument('--checkpoint-dir', type=str, help='Checkpoint directory')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model')
    eval_parser.add_argument('--model-path', type=str, required=True, help='Path to model file')
    eval_parser.add_argument('--config', type=str, help='Configuration file path')
    eval_parser.add_argument('--episodes', type=int, default=10, help='Number of evaluation episodes')
    
    # Visualize command
    vis_parser = subparsers.add_parser('visualize', help='Visualize training logs')
    vis_parser.add_argument('--log-path', type=str, required=True, help='Path to log file')
    vis_parser.add_argument('--save-path', type=str, help='Path to save plot')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run a quick demo')
    demo_parser.add_argument('--env', type=str, default='MiniGrid', help='Environment name')
    demo_parser.add_argument('--episodes', type=int, default=3, help='Number of demo episodes')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute command
    if args.command == 'train':
        train_command(args)
    elif args.command == 'evaluate':
        evaluate_command(args)
    elif args.command == 'visualize':
        visualize_command(args)
    elif args.command == 'demo':
        demo_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
