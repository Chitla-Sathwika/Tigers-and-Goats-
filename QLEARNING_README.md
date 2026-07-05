# Q-Learning Implementation for Huligutta (Goats and Tigers)

This document explains how to use the Q-learning algorithm implementation for the Huligutta game.

## Overview

The Q-learning implementation consists of three main components:

1. **qlearning.py** - The Q-learning agent class
2. **train_qlearning.py** - Training script for Q-learning agents
3. **game.py** - Updated game with Q-learning integration

## Quick Start

### 1. Train the Q-Learning Agents

First, train the agents by running the training script:

```bash
python train_qlearning.py
```

This will:
- Train a Tiger agent for 1000 episodes
- Train a Goat agent for 1000 episodes
- Save Q-tables as `q_table_tiger_final.pkl` and `q_table_goat_final.pkl`

### 2. Play Against Q-Learning Agent

After training, you can play against the trained agents:

```python
# In game.py, change the mode:
mode = 'qlearningTiger'  # You play as Goat, AI plays as Tiger
# or
mode = 'qlearningGoat'    # You play as Tiger, AI plays as Goat
# or
mode = 'qlearningBoth'    # Watch two AI agents play each other
```

Then run:
```bash
python game.py
```

## Training Parameters

You can customize training by modifying parameters in `train_qlearning.py`:

- **learning_rate** (default: 0.1) - How quickly the agent learns
- **discount_factor** (default: 0.9) - Importance of future rewards
- **epsilon** (default: 1.0) - Initial exploration rate
- **epsilon_decay** (default: 0.995) - How quickly exploration decreases
- **epsilon_min** (default: 0.01) - Minimum exploration rate
- **episodes** (default: 1000) - Number of training games

## How It Works

### State Representation

The Q-learning agent represents game states using:
- Tiger positions (tuple of sorted positions)
- Goat positions (tuple of sorted positions)

This creates a simplified state space that's manageable for Q-learning.

### Action Space

For Tigers:
- **move**: Move to an adjacent empty position
- **capture**: Jump over a goat to capture it

For Goats:
- **place**: Place a new goat on an empty position (when < 15 goats)
- **move**: Move to an adjacent empty position (when 15 goats placed)

### Rewards

- **Win**: +100
- **Loss**: -100
- **Capture (Tiger)**: +10
- **Blocking Tigers (Goat)**: Small positive reward based on limiting tiger moves

### Q-Learning Update

The Q-value is updated using the standard Q-learning formula:

```
Q(s,a) = Q(s,a) + α[r + γ * max(Q(s',a')) - Q(s,a)]
```

Where:
- `α` (alpha) = learning rate
- `γ` (gamma) = discount factor
- `r` = reward
- `s` = current state
- `s'` = next state
- `a` = action taken

## File Structure

```
├── qlearning.py          # Q-learning agent implementation
├── train_qlearning.py   # Training script
├── game.py             # Main game with Q-learning integration
├── q_table_tiger_*.pkl # Saved Q-tables for tiger agent
└── q_table_goat_*.pkl  # Saved Q-tables for goat agent
```

## Game Modes

### Available Modes in game.py:

1. **'pvp'** - Player vs Player
2. **'goatPlayer'** - You play as Goat, random AI as Tiger
3. **'tigerPlayer'** - You play as Tiger, random AI as Goat
4. **'qlearningTiger'** - You play as Goat, Q-learning agent as Tiger
5. **'qlearningGoat'** - You play as Tiger, Q-learning agent as Goat
6. **'qlearningBoth'** - Both players are Q-learning agents (auto-play)

## Tips for Training

1. **More episodes = better performance**: Train for at least 1000 episodes for decent play
2. **Adjust learning rate**: Lower learning rates (0.01-0.05) for more stable learning
3. **Monitor win rate**: Check the training output to see if agents are improving
4. **Save frequently**: Q-tables are saved every 100 episodes by default

## Troubleshooting

### Q-table file not found
If you see "Warning: Q-table file not found", make sure you've trained the agents first by running `train_qlearning.py`.

### Poor performance
- Try training for more episodes
- Adjust learning rate and discount factor
- Check that the game rules are being followed correctly

### Import errors
Make sure all dependencies are installed:
```bash
pip install numpy networkx pillow
```

## Future Improvements

Potential enhancements:
- Deep Q-Network (DQN) for larger state spaces
- Self-play training (agents learn by playing each other)
- Reward shaping for better learning
- State abstraction to reduce state space
- Multi-step lookahead

## References

- Q-Learning algorithm: Reinforcement Learning by Sutton & Barto
- Game rules: Traditional Huligutta/Bagh-Chal rules

