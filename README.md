# Distributional Reinforcement Learning

A well-structured implementation of distributional reinforcement learning algorithms, featuring state-of-the-art techniques like C51 (Categorical DQN) and comprehensive tooling for training, evaluation, and visualization.

## Features

- **Modern Distributional RL**: Implementation of C51 (Categorical DQN) algorithm
- **Multiple Environments**: Support for custom MiniGridWorld and gymnasium environments
- **Comprehensive Tooling**: CLI interface, configuration management, logging, and visualization
- **Type Safety**: Full type hints and comprehensive docstrings
- **Testing**: Unit tests for all core components
- **Interactive Demo**: Jupyter notebook for exploration and experimentation
- **Reproducible**: Seed management and checkpoint saving/loading

## 📁 Project Structure

```
├── src/
│   ├── agents/          # RL agent implementations
│   ├── envs/            # Environment implementations
│   └── utils/           # Utilities, logging, visualization
├── config/              # Configuration files
├── notebooks/           # Jupyter notebooks
├── tests/               # Unit tests
├── logs/                # Training logs
├── checkpoints/         # Model checkpoints
├── train.py             # Main training script
├── cli.py               # Command-line interface
├── requirements.txt     # Dependencies
└── README.md            # This file
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kryptologyst/Distributional-Reinforcement-Learning.git
   cd Distributional-Reinforcement-Learning
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python -c "import torch; import gymnasium; print('Installation successful!')"
   ```

## Quick Start

### Command Line Interface

The easiest way to get started is using the CLI:

```bash
# Run a quick demo
python cli.py demo --env MiniGrid --episodes 3

# Train an agent
python cli.py train --env MiniGrid --episodes 500

# Train on CartPole
python cli.py train --env CartPole-v1 --episodes 1000

# Evaluate a trained model
python cli.py evaluate --model-path checkpoints/model.pt --episodes 10

# Visualize training logs
python cli.py visualize --log-path logs/training.json
```

### Python API

```python
import sys
from pathlib import Path
sys.path.append(str(Path.cwd() / "src"))

from envs import make_env
from agents import C51Agent
from utils import set_seed

# Set seed for reproducibility
set_seed(42)

# Create environment
env = make_env("MiniGrid")

# Create agent
agent = C51Agent(state_dim=2, action_dim=4)

# Training loop
for episode in range(100):
    state, _ = env.reset()
    done = False
    
    while not done:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        agent.store_transition(state, action, reward, next_state, done)
        agent.train()
        
        state = next_state
```

### Jupyter Notebook

For interactive exploration, use the provided notebook:

```bash
jupyter notebook notebooks/distributional_rl_demo.ipynb
```

## 🔧 Configuration

The project uses YAML configuration files. Create custom configurations by copying `config/default.yaml`:

```yaml
# config/custom.yaml
default_config:
  env:
    name: "CartPole-v1"
  agent:
    algorithm: "C51"
    lr: 0.0005
    atoms: 51
  training:
    episodes: 2000
```

Use custom configurations:

```bash
python cli.py train --config config/custom.yaml
```

## Testing

Run the test suite to verify everything works correctly:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_components.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

## Visualization

The project includes comprehensive visualization tools:

- **Learning Curves**: Episode rewards, lengths, and training metrics
- **Distribution Plots**: Return distributions for different actions
- **Q-value Heatmaps**: Action-value function visualization
- **Policy Evaluation**: Step-by-step policy execution

## Architecture

### Agents

- **C51Agent**: Categorical DQN implementation with neural networks
- **DistributionalDQN**: Neural network for distributional Q-learning
- **ReplayBuffer**: Experience replay for stable training

### Environments

- **MiniGridWorld**: Custom 3x3 grid world environment
- **DistributionalEnvWrapper**: Wrapper for gymnasium environments
- **Environment Factory**: Easy environment creation and management

### Utilities

- **MetricsLogger**: Comprehensive training metrics tracking
- **Visualizer**: Plotting and visualization tools
- **Config**: YAML configuration management
- **Checkpointing**: Model saving and loading

## Algorithm Details

### C51 (Categorical DQN)

C51 learns the full distribution of returns instead of just expected values:

1. **Distributional Bellman Update**: Projects target distributions onto fixed support
2. **Categorical Projection**: Maintains probability distributions over returns
3. **KL Divergence Loss**: Minimizes divergence between current and target distributions
4. **Risk-Sensitive Decision Making**: Captures uncertainty and risk preferences

### Key Advantages

- **Uncertainty Quantification**: Provides full distribution information
- **Risk Sensitivity**: Enables risk-aware decision making
- **Better Sample Efficiency**: Often learns faster than standard DQN
- **Robustness**: More stable training in stochastic environments

## Performance

The implementation achieves strong performance on standard benchmarks:

- **MiniGridWorld**: Consistently reaches goal in <20 steps
- **CartPole-v1**: Solves environment (avg reward >475) in ~500 episodes
- **MountainCar-v0**: Efficient exploration and goal reaching

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## References

- [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887) - Bellemare et al.
- [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298) - Hessel et al.
- [Distributional Reinforcement Learning with Quantile Regression](https://arxiv.org/abs/1710.10044) - Dabney et al.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI Gym/Gymnasium for environment interfaces
- PyTorch for deep learning framework
- The RL research community for foundational algorithms

 
# Distributional-Reinforcement-Learning
