"""
Unit tests for distributional RL components.

This module contains tests for environments, agents, and utilities.
"""

import unittest
import numpy as np
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from envs import MiniGridWorld, DistributionalEnvWrapper, make_env
from agents import C51Agent, DistributionalDQN, ReplayBuffer
from utils import MetricsLogger, Visualizer, set_seed
from utils.config import Config


class TestEnvironments(unittest.TestCase):
    """Test environment implementations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.env = MiniGridWorld()
    
    def test_minigrid_reset(self):
        """Test MiniGridWorld reset functionality."""
        obs, info = self.env.reset()
        self.assertEqual(obs.tolist(), [0, 0])
        self.assertEqual(info, {})
    
    def test_minigrid_step(self):
        """Test MiniGridWorld step functionality."""
        obs, _ = self.env.reset()
        
        # Test valid action
        next_obs, reward, terminated, truncated, info = self.env.step(1)  # Down
        self.assertEqual(next_obs.tolist(), [1, 0])
        self.assertEqual(reward, 0.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
    
    def test_minigrid_goal_reaching(self):
        """Test reaching the goal in MiniGridWorld."""
        obs, _ = self.env.reset()
        
        # Move to goal position (2, 2)
        obs, _, _, _, _ = self.env.step(1)  # Down
        obs, _, _, _, _ = self.env.step(1)  # Down
        obs, _, _, _, _ = self.env.step(3)  # Right
        obs, _, _, _, _ = self.env.step(3)  # Right
        
        # Should be at goal
        self.assertEqual(obs.tolist(), [2, 2])
        
        # Next step should give reward and terminate
        obs, reward, terminated, truncated, info = self.env.step(0)  # Any action
        self.assertEqual(reward, 1.0)
        self.assertTrue(terminated)
    
    def test_state_to_index(self):
        """Test state to index conversion."""
        self.assertEqual(self.env.state_to_index((0, 0)), 0)
        self.assertEqual(self.env.state_to_index((1, 1)), 4)
        self.assertEqual(self.env.state_to_index((2, 2)), 8)
    
    def test_make_env(self):
        """Test environment factory function."""
        env = make_env("MiniGrid")
        self.assertIsInstance(env, MiniGridWorld)
        
        # Test with gymnasium environment
        try:
            env = make_env("CartPole-v1")
            self.assertIsInstance(env, DistributionalEnvWrapper)
        except Exception:
            # Skip if gymnasium not available
            pass


class TestAgents(unittest.TestCase):
    """Test agent implementations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.state_dim = 4
        self.action_dim = 2
        self.agent = C51Agent(self.state_dim, self.action_dim)
    
    def test_c51_initialization(self):
        """Test C51 agent initialization."""
        self.assertEqual(self.agent.state_dim, self.state_dim)
        self.assertEqual(self.agent.action_dim, self.action_dim)
        self.assertIsNotNone(self.agent.q_net)
        self.assertIsNotNone(self.agent.target_net)
        self.assertIsNotNone(self.agent.optimizer)
        self.assertIsNotNone(self.agent.replay_buffer)
    
    def test_action_selection(self):
        """Test action selection."""
        state = np.random.randn(self.state_dim)
        action = self.agent.select_action(state)
        
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.action_dim)
    
    def test_transition_storage(self):
        """Test transition storage."""
        state = np.random.randn(self.state_dim)
        action = 0
        reward = 1.0
        next_state = np.random.randn(self.state_dim)
        done = False
        
        initial_size = len(self.agent.replay_buffer)
        self.agent.store_transition(state, action, reward, next_state, done)
        
        self.assertEqual(len(self.agent.replay_buffer), initial_size + 1)
    
    def test_training(self):
        """Test training functionality."""
        # Add some transitions to buffer
        for _ in range(50):  # More than batch_size
            state = np.random.randn(self.state_dim)
            action = np.random.randint(self.action_dim)
            reward = np.random.randn()
            next_state = np.random.randn(self.state_dim)
            done = np.random.random() < 0.1
            
            self.agent.store_transition(state, action, reward, next_state, done)
        
        # Test training
        loss = self.agent.train()
        self.assertIsNotNone(loss)
        self.assertIsInstance(loss, float)
    
    def test_distributional_dqn(self):
        """Test DistributionalDQN network."""
        net = DistributionalDQN(self.state_dim, self.action_dim)
        
        # Test forward pass
        x = torch.randn(1, self.state_dim)
        output = net(x)
        
        self.assertEqual(output.shape, (1, self.action_dim, net.atoms))
        
        # Test Q-values
        q_values = net.get_q_values(x)
        self.assertEqual(q_values.shape, (1, self.action_dim))
    
    def test_replay_buffer(self):
        """Test replay buffer functionality."""
        buffer = ReplayBuffer(capacity=10)
        
        # Test adding transitions
        for i in range(15):  # More than capacity
            buffer.push(
                np.array([i]), i % 2, float(i), np.array([i + 1]), i % 3 == 0
            )
        
        # Should only keep last 10
        self.assertEqual(len(buffer), 10)
        
        # Test sampling
        batch = buffer.sample(5)
        self.assertEqual(len(batch), 5)
        self.assertEqual(len(batch[0]), 5)  # batch_size


class TestUtils(unittest.TestCase):
    """Test utility functions."""
    
    def test_metrics_logger(self):
        """Test metrics logger."""
        logger = MetricsLogger()
        
        # Test metric logging
        logger.log_metric("test_metric", 1.0, 0)
        self.assertEqual(len(logger.metrics["test_metric"]), 1)
        
        # Test episode logging
        logger.log_episode(10.0, 100, 0)
        self.assertEqual(len(logger.episode_rewards), 1)
        self.assertEqual(len(logger.episode_lengths), 1)
        
        # Test summary
        summary = logger.get_summary()
        self.assertIn("total_episodes", summary)
        self.assertEqual(summary["total_episodes"], 1)
    
    def test_seed_setting(self):
        """Test random seed setting."""
        set_seed(42)
        
        # Test numpy
        np.random.seed(42)
        val1 = np.random.randn()
        
        set_seed(42)
        val2 = np.random.randn()
        
        self.assertEqual(val1, val2)
    
    def test_config(self):
        """Test configuration management."""
        # Create a temporary config
        config = Config()
        
        # Test setting and getting values
        config.set("test.value", 42)
        self.assertEqual(config.get("test.value"), 42)
        
        # Test default values
        self.assertEqual(config.get("nonexistent.key", "default"), "default")
        
        # Test dictionary-style access
        config["test.value2"] = 24
        self.assertEqual(config["test.value2"], 24)


class TestIntegration(unittest.TestCase):
    """Integration tests."""
    
    def test_training_loop(self):
        """Test a simple training loop."""
        env = MiniGridWorld()
        agent = C51Agent(state_dim=2, action_dim=4)
        
        # Run a few episodes
        for episode in range(5):
            state, _ = env.reset()
            done = False
            
            while not done:
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                agent.store_transition(state, action, reward, next_state, done)
                agent.train()
                
                state = next_state
        
        # Agent should have learned something
        self.assertGreater(len(agent.replay_buffer), 0)
        self.assertGreater(len(agent.losses), 0)


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
