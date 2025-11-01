"""
Modern distributional reinforcement learning agents.

This module implements state-of-the-art distributional RL algorithms including
C51, QR-DQN, and other distributional variants.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random


class DistributionalDQN(nn.Module):
    """
    Neural network for distributional DQN (C51-style).
    
    Outputs probability distributions over returns instead of single Q-values.
    """
    
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        atoms: int = 51,
        hidden_dim: int = 128,
        v_min: float = -10.0,
        v_max: float = 10.0
    ):
        """
        Initialize the distributional DQN network.
        
        Args:
            input_dim: Dimension of input state
            action_dim: Number of actions
            atoms: Number of atoms in the distribution
            hidden_dim: Hidden layer dimension
            v_min: Minimum value for distribution support
            v_max: Maximum value for distribution support
        """
        super().__init__()
        self.atoms = atoms
        self.action_dim = action_dim
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (atoms - 1)
        self.z = torch.linspace(v_min, v_max, atoms)
        
        # Network layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim * atoms)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input state tensor
            
        Returns:
            Probability distributions for each action
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        # Reshape to (batch_size, action_dim, atoms)
        x = x.view(-1, self.action_dim, self.atoms)
        
        # Apply softmax to get probabilities
        return F.softmax(x, dim=-1)
    
    def get_q_values(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get Q-values by computing expected values from distributions.
        
        Args:
            x: Input state tensor
            
        Returns:
            Q-values for each action
        """
        distributions = self.forward(x)
        q_values = torch.sum(distributions * self.z, dim=-1)
        return q_values


class ReplayBuffer:
    """
    Experience replay buffer for distributional RL.
    
    Stores transitions and supports sampling for training.
    """
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize the replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool) -> None:
        """
        Add a transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        Sample a batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Batch of transitions
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)


class C51Agent:
    """
    C51 (Categorical DQN) agent implementation.
    
    Learns the full distribution of returns instead of just expected values.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        batch_size: int = 32,
        buffer_size: int = 10000,
        target_update_freq: int = 100
    ):
        """
        Initialize the C51 agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions
            lr: Learning rate
            gamma: Discount factor
            epsilon: Exploration rate
            epsilon_decay: Epsilon decay rate
            epsilon_min: Minimum epsilon
            atoms: Number of atoms in distribution
            v_min: Minimum value for distribution support
            v_max: Maximum value for distribution support
            batch_size: Batch size for training
            buffer_size: Size of replay buffer
            target_update_freq: Frequency of target network updates
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.atoms = atoms
        self.v_min = v_min
        self.v_max = v_max
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Initialize networks
        self.q_net = DistributionalDQN(
            state_dim, action_dim, atoms, v_min=v_min, v_max=v_max
        )
        self.target_net = DistributionalDQN(
            state_dim, action_dim, atoms, v_min=v_min, v_max=v_max
        )
        
        # Copy weights to target network
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        # Initialize optimizer and replay buffer
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Training statistics
        self.step_count = 0
        self.losses = []
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Selected action
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_net.get_q_values(state_tensor)
            return q_values.argmax().item()
    
    def store_transition(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool) -> None:
        """
        Store a transition in the replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train(self) -> Optional[float]:
        """
        Train the agent on a batch of experiences.
        
        Returns:
            Training loss if training occurred, None otherwise
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)
        
        # Get current distributions
        current_distributions = self.q_net(states)
        current_distributions = current_distributions[range(self.batch_size), actions]
        
        # Compute target distributions
        with torch.no_grad():
            next_distributions = self.target_net(next_states)
            next_q_values = torch.sum(next_distributions * self.q_net.z, dim=-1)
            next_actions = next_q_values.argmax(dim=1)
            next_distributions = next_distributions[range(self.batch_size), next_actions]
            
            # Compute target atoms
            target_atoms = rewards.unsqueeze(1) + self.gamma * self.q_net.z.unsqueeze(0) * (~dones).unsqueeze(1)
            target_atoms = torch.clamp(target_atoms, self.v_min, self.v_max)
            
            # Project onto support
            target_distributions = self._project_distribution(
                next_distributions, target_atoms
            )
        
        # Compute loss
        loss = -torch.sum(target_distributions * torch.log(current_distributions + 1e-8), dim=1)
        loss = loss.mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.losses.append(loss.item())
        return loss.item()
    
    def _project_distribution(self, distributions: torch.Tensor, 
                            target_atoms: torch.Tensor) -> torch.Tensor:
        """
        Project target distribution onto the support atoms.
        
        Args:
            distributions: Source distributions
            target_atoms: Target atom values
            
        Returns:
            Projected distributions
        """
        batch_size = distributions.size(0)
        atoms = self.q_net.atoms
        
        # Compute projection
        delta_z = self.q_net.delta_z
        b = (target_atoms - self.v_min) / delta_z
        l = torch.floor(b).long()
        u = torch.ceil(b).long()
        
        # Clamp indices
        l = torch.clamp(l, 0, atoms - 1)
        u = torch.clamp(u, 0, atoms - 1)
        
        # Compute projection weights
        weights = torch.zeros_like(distributions)
        weights.scatter_add_(1, l, distributions * (u.float() - b))
        weights.scatter_add_(1, u, distributions * (b - l.float()))
        
        return weights
    
    def save(self, filepath: str) -> None:
        """Save the agent's model."""
        torch.save({
            'q_net_state_dict': self.q_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count
        }, filepath)
    
    def load(self, filepath: str) -> None:
        """Load the agent's model."""
        checkpoint = torch.load(filepath)
        self.q_net.load_state_dict(checkpoint['q_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']


if __name__ == "__main__":
    # Test the C51 agent
    print("Testing C51 Agent...")
    
    # Create a simple test environment
    state_dim = 4
    action_dim = 2
    
    agent = C51Agent(state_dim, action_dim)
    
    # Test action selection
    state = np.random.randn(state_dim)
    action = agent.select_action(state)
    print(f"Selected action: {action}")
    
    # Test training
    for _ in range(100):
        next_state = np.random.randn(state_dim)
        reward = np.random.randn()
        done = np.random.random() < 0.1
        
        agent.store_transition(state, action, reward, next_state, done)
        loss = agent.train()
        
        if loss is not None:
            print(f"Training loss: {loss:.4f}")
            break
        
        state = next_state
        action = agent.select_action(state)
    
    print("C51 Agent test completed!")
