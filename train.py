#!/usr/bin/env python3
"""
Main training script for distributional reinforcement learning.

This script provides a command-line interface for training distributional RL agents
on various environments.
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import numpy as np
import torch
from tqdm import tqdm

from envs import make_env
from agents import C51Agent
from utils import MetricsLogger, Visualizer, set_seed, create_checkpoint_dir
from utils.config import load_config


def train_agent(agent, env, config, logger, visualizer, checkpoint_dir):
    """
    Train the distributional RL agent.
    
    Args:
        agent: The RL agent to train
        env: The environment
        config: Configuration object
        logger: Metrics logger
        visualizer: Visualization utilities
        checkpoint_dir: Directory to save checkpoints
    """
    episodes = config.get('training.episodes', 1000)
    max_steps = config.get('training.max_steps_per_episode', 1000)
    eval_freq = config.get('training.eval_freq', 100)
    save_freq = config.get('training.save_freq', 500)
    log_freq = config.get('training.log_freq', 10)
    plot_freq = config.get('logging.plot_freq', 100)
    
    print(f"Starting training for {episodes} episodes...")
    
    for episode in tqdm(range(episodes), desc="Training"):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(max_steps):
            # Select action
            action = agent.select_action(state, training=True)
            
            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Train agent
            loss = agent.train()
            
            # Update statistics
            episode_reward += reward
            episode_length += 1
            
            # Log metrics
            if loss is not None:
                logger.log_metric("loss", loss)
                logger.log_metric("epsilon", agent.epsilon)
            
            state = next_state
            
            if done:
                break
        
        # Log episode statistics
        logger.log_episode(episode_reward, episode_length)
        
        # Print progress
        if episode % log_freq == 0:
            avg_reward = np.mean(list(logger.episode_rewards)[-10:]) if len(logger.episode_rewards) >= 10 else episode_reward
            print(f"Episode {episode}: Reward={episode_reward:.2f}, "
                  f"Avg Reward (10)={avg_reward:.2f}, "
                  f"Epsilon={agent.epsilon:.3f}")
        
        # Evaluation
        if episode % eval_freq == 0 and episode > 0:
            eval_reward = evaluate_agent(agent, env, num_episodes=5)
            logger.log_metric("eval_reward", eval_reward, episode)
            print(f"Evaluation at episode {episode}: Avg Reward={eval_reward:.2f}")
        
        # Save checkpoints
        if episode % save_freq == 0 and episode > 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"agent_episode_{episode}.pt")
            agent.save(checkpoint_path)
            print(f"Saved checkpoint at episode {episode}")
        
        # Plot learning curves
        if episode % plot_freq == 0 and episode > 0:
            plot_path = os.path.join(checkpoint_dir, f"learning_curves_episode_{episode}.png")
            visualizer.plot_learning_curves(logger, save_path=plot_path)
    
    # Final evaluation
    print("Training completed! Running final evaluation...")
    final_eval_reward = evaluate_agent(agent, env, num_episodes=10)
    logger.log_metric("final_eval_reward", final_eval_reward)
    print(f"Final evaluation reward: {final_eval_reward:.2f}")
    
    # Save final model and logs
    final_checkpoint_path = os.path.join(checkpoint_dir, "final_model.pt")
    agent.save(final_checkpoint_path)
    logger.save_logs()
    
    # Final plots
    final_plot_path = os.path.join(checkpoint_dir, "final_learning_curves.png")
    visualizer.plot_learning_curves(logger, save_path=final_plot_path)
    
    print(f"Training completed! Checkpoints saved to: {checkpoint_dir}")


def evaluate_agent(agent, env, num_episodes=10):
    """
    Evaluate the agent's performance.
    
    Args:
        agent: The RL agent
        env: The environment
        num_episodes: Number of episodes to evaluate
        
    Returns:
        Average reward over evaluation episodes
    """
    total_rewards = []
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        while True:
            action = agent.select_action(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
            
            state = next_state
        
        total_rewards.append(episode_reward)
    
    return np.mean(total_rewards)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train distributional RL agents")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--env", type=str, help="Environment name")
    parser.add_argument("--episodes", type=int, help="Number of training episodes")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--checkpoint-dir", type=str, help="Checkpoint directory")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.env:
        config.set('env.name', args.env)
    if args.episodes:
        config.set('training.episodes', args.episodes)
    if args.seed:
        config.set('seed', args.seed)
    
    # Set random seed
    seed = config.get('seed', 42)
    set_seed(seed)
    print(f"Set random seed to {seed}")
    
    # Create checkpoint directory
    checkpoint_dir = args.checkpoint_dir or create_checkpoint_dir()
    print(f"Checkpoint directory: {checkpoint_dir}")
    
    # Create environment
    env_name = config.get('env.name', 'MiniGrid')
    env_kwargs = {}
    
    if env_name == 'MiniGrid':
        env_kwargs['size'] = config.get('env.size', 3)
    
    env = make_env(env_name, **env_kwargs)
    print(f"Created environment: {env_name}")
    
    # Get environment dimensions
    if hasattr(env, 'observation_space'):
        if hasattr(env.observation_space, 'shape'):
            state_dim = env.observation_space.shape[0]
        else:
            state_dim = env.observation_space.n
    else:
        state_dim = config.get('agent.state_dim', 2)
    
    action_dim = env.action_space.n if hasattr(env.action_space, 'n') else config.get('agent.action_dim', 4)
    
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
    
    # Update config with actual dimensions
    config.set('agent.state_dim', state_dim)
    config.set('agent.action_dim', action_dim)
    
    # Create agent
    algorithm = config.get('agent.algorithm', 'C51')
    
    if algorithm == 'C51':
        agent = C51Agent(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=config.get('agent.lr', 0.001),
            gamma=config.get('agent.gamma', 0.99),
            epsilon=config.get('agent.epsilon', 0.1),
            epsilon_decay=config.get('agent.epsilon_decay', 0.995),
            epsilon_min=config.get('agent.epsilon_min', 0.01),
            atoms=config.get('agent.atoms', 51),
            v_min=config.get('agent.v_min', -10.0),
            v_max=config.get('agent.v_max', 10.0),
            batch_size=config.get('agent.batch_size', 32),
            buffer_size=config.get('agent.buffer_size', 10000),
            target_update_freq=config.get('agent.target_update_freq', 100)
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    print(f"Created {algorithm} agent")
    
    # Create logger and visualizer
    logger = MetricsLogger(checkpoint_dir)
    visualizer = Visualizer()
    
    # Print configuration
    print("\nConfiguration:")
    print(f"  Environment: {env_name}")
    print(f"  Algorithm: {algorithm}")
    print(f"  Episodes: {config.get('training.episodes', 1000)}")
    print(f"  Learning rate: {config.get('agent.lr', 0.001)}")
    print(f"  Gamma: {config.get('agent.gamma', 0.99)}")
    print(f"  Atoms: {config.get('agent.atoms', 51)}")
    print(f"  V_min: {config.get('agent.v_min', -10.0)}")
    print(f"  V_max: {config.get('agent.v_max', 10.0)}")
    print()
    
    # Start training
    try:
        train_agent(agent, env, config, logger, visualizer, checkpoint_dir)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        # Save current state
        checkpoint_path = os.path.join(checkpoint_dir, "interrupted_model.pt")
        agent.save(checkpoint_path)
        logger.save_logs()
        print(f"Saved interrupted state to: {checkpoint_path}")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
