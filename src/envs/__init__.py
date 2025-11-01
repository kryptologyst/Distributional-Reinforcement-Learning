"""
Environment implementations for distributional reinforcement learning.

This module provides both custom environments and wrappers for gymnasium environments
to support distributional RL algorithms.
"""

from typing import Tuple, Optional, Any, Dict
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt


class MiniGridWorld(gym.Env):
    """
    A simple 3x3 grid world environment for testing distributional RL algorithms.
    
    The agent starts at (0,0) and must reach the goal at (2,2).
    Actions: 0=Up, 1=Down, 2=Left, 3=Right
    """
    
    def __init__(self, size: int = 3, goal_pos: Optional[Tuple[int, int]] = None):
        """
        Initialize the MiniGridWorld environment.
        
        Args:
            size: Size of the grid (default: 3)
            goal_pos: Position of the goal (default: (size-1, size-1))
        """
        super().__init__()
        self.size = size
        self.goal = goal_pos if goal_pos is not None else (size - 1, size - 1)
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space = spaces.Box(
            low=0, high=size-1, shape=(2,), dtype=np.int32
        )
        
        # Action mappings
        self.action_mapping = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # U, D, L, R
        
        self.reset()
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        self.agent_pos = np.array([0, 0], dtype=np.int32)
        return self.agent_pos.copy(), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take (0-3)
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Move agent
        dr, dc = self.action_mapping[action]
        new_r = np.clip(self.agent_pos[0] + dr, 0, self.size - 1)
        new_c = np.clip(self.agent_pos[1] + dc, 0, self.size - 1)
        self.agent_pos = np.array([new_r, new_c], dtype=np.int32)
        
        # Calculate reward and done status
        reward = 1.0 if tuple(self.agent_pos) == self.goal else 0.0
        terminated = tuple(self.agent_pos) == self.goal
        truncated = False
        
        return self.agent_pos.copy(), reward, terminated, truncated, {}
    
    def state_to_index(self, pos: Tuple[int, int]) -> int:
        """Convert position to state index."""
        return pos[0] * self.size + pos[1]
    
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render the environment."""
        if mode == "human":
            grid = np.zeros((self.size, self.size))
            grid[self.agent_pos[0], self.agent_pos[1]] = 1  # Agent
            grid[self.goal[0], self.goal[1]] = 2  # Goal
            
            plt.figure(figsize=(4, 4))
            plt.imshow(grid, cmap='viridis')
            plt.title("MiniGridWorld")
            plt.xticks(range(self.size))
            plt.yticks(range(self.size))
            plt.show()
        
        return None


class DistributionalEnvWrapper(gym.Wrapper):
    """
    Wrapper for gymnasium environments to add distributional RL support.
    
    This wrapper adds methods and properties useful for distributional RL algorithms.
    """
    
    def __init__(self, env: gym.Env, reward_scale: float = 1.0):
        """
        Initialize the wrapper.
        
        Args:
            env: The gymnasium environment to wrap
            reward_scale: Scale factor for rewards
        """
        super().__init__(env)
        self.reward_scale = reward_scale
        self._episode_rewards = []
        self._episode_length = 0
    
    def reset(self, **kwargs) -> Tuple[Any, Dict]:
        """Reset the environment and clear episode statistics."""
        obs, info = self.env.reset(**kwargs)
        self._episode_rewards = []
        self._episode_length = 0
        return obs, info
    
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict]:
        """Step the environment and track episode statistics."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Scale reward
        reward *= self.reward_scale
        
        # Track episode statistics
        self._episode_rewards.append(reward)
        self._episode_length += 1
        
        # Add episode statistics to info
        info["episode_reward"] = sum(self._episode_rewards)
        info["episode_length"] = self._episode_length
        
        return obs, reward, terminated, truncated, info
    
    @property
    def episode_reward(self) -> float:
        """Get the total reward for the current episode."""
        return sum(self._episode_rewards)
    
    @property
    def episode_length(self) -> int:
        """Get the length of the current episode."""
        return self._episode_length


def make_env(env_name: str, **kwargs) -> gym.Env:
    """
    Create an environment by name.
    
    Args:
        env_name: Name of the environment
        **kwargs: Additional arguments for environment creation
        
    Returns:
        The created environment
    """
    if env_name == "MiniGrid":
        return MiniGridWorld(**kwargs)
    elif env_name in ["CartPole-v1", "MountainCar-v0", "Acrobot-v1"]:
        env = gym.make(env_name, **kwargs)
        return DistributionalEnvWrapper(env)
    else:
        raise ValueError(f"Unknown environment: {env_name}")


if __name__ == "__main__":
    # Test the environments
    print("Testing MiniGridWorld...")
    env = MiniGridWorld()
    obs, _ = env.reset()
    print(f"Initial observation: {obs}")
    
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i}: action={action}, obs={obs}, reward={reward}, done={terminated}")
        if terminated:
            break
    
    print("\nTesting CartPole with wrapper...")
    env = make_env("CartPole-v1")
    obs, _ = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    for i in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i}: reward={reward:.3f}, episode_reward={env.episode_reward:.3f}")
        if terminated:
            break
