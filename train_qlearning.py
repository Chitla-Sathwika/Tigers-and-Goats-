'''
Q-Learning Training Script for Huligutta
file: train_qlearning.py
Description: Training script for Q-learning agents
'''

__author__ = "Clyde James Felix"
__email__ = "cjfelix.hawaii.edu"
__status__ = "Dev"

import random
from huligutta import *
from functions import *
from qlearning import QLearningAgent
from copy import deepcopy
import time

class GameEnvironment:
    """
    Game environment for training Q-learning agents.
    Handles game logic without GUI.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset the game to initial state."""
        Board().clearBoard()
        
        # Place 3 tigers at initial positions
        Tiger('b0').place()
        Tiger('c1').place()
        Tiger('d1').place()
        
        self.goat_count = 0
        self.goats_eaten = 0
        self.turn = False  # False = Goat turn, True = Tiger turn
        self.move_count = 0
        self.game_over = False
        self.winner = None
        
        return self.get_state()
    
    def update_goat_count(self):
        """Update goat count from board state."""
        self.goat_count = len(goatPositions(Board().boardPositions))
    
    def get_state(self):
        """Get current board state."""
        return deepcopy(Board().boardPositions)
    
    def is_game_over(self):
        """Check if game is over."""
        if self.game_over:
            return True
        
        # Check if tigers can't move (goats win)
        tigers = tigerPositions(Board().boardPositions)
        total_moves = sum(len(Tiger(t).possibleMoves()) for t in tigers)
        if total_moves == 0:
            self.game_over = True
            self.winner = 'goat'
            return True
        
        # Check if 5 goats eaten (tigers win)
        # Calculate goats eaten: started with 0, can place up to 15, so eaten = 15 - current_goats
        current_goats = len(goatPositions(Board().boardPositions))
        goats_eaten = 15 - current_goats
        if goats_eaten >= 5:
            self.game_over = True
            self.winner = 'tiger'
            self.goats_eaten = goats_eaten
            return True
        
        return False
    
    def execute_action(self, action, is_tiger_turn):
        """
        Execute an action and return new state and reward.
        
        Args:
            action: (action_type, from_pos, to_pos)
            is_tiger_turn: Whether it's tiger's turn
            
        Returns:
            (new_state, reward, done, info)
        """
        if action is None:
            return self.get_state(), -10, True, {'error': 'No valid action'}
        
        action_type, from_pos, to_pos = action
        
        try:
            if is_tiger_turn:
                if action_type == 'capture':
                    tiger = Tiger(from_pos)
                    if tiger.capture(to_pos) == 1:
                        self.goats_eaten += 1
                        self.turn = False
                        self.move_count += 1
                elif action_type == 'move':
                    tiger = Tiger(from_pos)
                    if tiger.move(to_pos) == 1:
                        self.turn = False
                        self.move_count += 1
                else:
                    return self.get_state(), -10, False, {'error': 'Invalid tiger action'}
            else:
                if action_type == 'place':
                    self.update_goat_count()
                    if self.goat_count < 15:
                        goat = Goat(to_pos)
                        goat.place()
                        self.update_goat_count()
                        self.turn = True
                    else:
                        return self.get_state(), -10, False, {'error': 'Cannot place more goats'}
                elif action_type == 'move':
                    self.update_goat_count()
                    if self.goat_count == 15:
                        goat = Goat(from_pos)
                        if goat.move(to_pos) == 1:
                            self.turn = True
                            self.move_count += 1
                    else:
                        return self.get_state(), -10, False, {'error': 'Must place all goats first'}
                else:
                    return self.get_state(), -10, False, {'error': 'Invalid goat action'}
            
            # Check if game is over
            done = self.is_game_over()
            
            # Update goat count after action
            self.update_goat_count()
            
            # Calculate reward
            reward = 0
            if done:
                if self.winner == 'tiger':
                    reward = 100
                elif self.winner == 'goat':
                    reward = -100
            else:
                # Small intermediate rewards
                if is_tiger_turn and action_type == 'capture':
                    reward = 10  # Reward for capturing
                elif not is_tiger_turn:
                    # Reward for blocking tigers
                    tigers = tigerPositions(Board().boardPositions)
                    total_moves = sum(len(Tiger(t).possibleMoves()) for t in tigers)
                    reward = (10 - total_moves) * 0.5
            
            return self.get_state(), reward, done, {}
            
        except Exception as e:
            return self.get_state(), -10, False, {'error': str(e)}


def train_agent(agent, episodes=1000, opponent='random', save_interval=100):
    """
    Train Q-learning agent.
    
    Args:
        agent: QLearningAgent instance
        episodes: Number of training episodes
        opponent: 'random' or another QLearningAgent
        save_interval: Save Q-table every N episodes
    """
    print(f"Starting training for {agent.player} agent...")
    print(f"Episodes: {episodes}, Learning rate: {agent.learning_rate}, "
          f"Discount: {agent.discount_factor}, Epsilon: {agent.epsilon}")
    
    wins = 0
    losses = 0
    
    for episode in range(episodes):
        env = GameEnvironment()
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        # Determine if agent plays first
        agent_turn = (agent.player == 'goat')
        
        while not done and steps < 500:  # Max steps per episode
            steps += 1
            
            if agent_turn:
                # Agent's turn
                action = agent.choose_action(state, agent.player == 'tiger', training=True)
                next_state, reward, done, info = env.execute_action(action, agent.player == 'tiger')
                
                # Adjust reward based on agent's perspective
                if agent.player == 'goat':
                    reward = -reward  # Invert reward for goats
                
                # Update Q-value
                agent.update_q_value(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
                agent_turn = False
            else:
                # Opponent's turn
                if opponent == 'random':
                    # Random opponent
                    actions = agent.get_actions(state, not (agent.player == 'tiger'))
                    if actions:
                        action = random.choice(actions)
                        next_state, reward, done, info = env.execute_action(
                            action, not (agent.player == 'tiger')
                        )
                        state = next_state
                else:
                    # Another Q-learning agent as opponent
                    action = opponent.choose_action(state, opponent.player == 'tiger', training=True)
                    next_state, reward, done, info = env.execute_action(action, opponent.player == 'tiger')
                    state = next_state
                
                agent_turn = True
            
            if done:
                break
        
        # Update statistics
        if done:
            if env.winner == agent.player:
                wins += 1
                agent.training_stats['wins'] += 1
            else:
                losses += 1
                agent.training_stats['losses'] += 1
        
        agent.training_stats['episodes'] += 1
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Print progress
        if (episode + 1) % 100 == 0:
            win_rate = wins / (episode + 1) * 100
            print(f"Episode {episode + 1}/{episodes} - "
                  f"Win rate: {win_rate:.1f}% - "
                  f"Epsilon: {agent.epsilon:.3f} - "
                  f"Q-table size: {len(agent.q_table)}")
        
        # Save Q-table periodically
        if (episode + 1) % save_interval == 0:
            filename = f"q_table_{agent.player}_{episode + 1}.pkl"
            agent.save_q_table(filename)
    
    # Final save
    filename = f"q_table_{agent.player}_final.pkl"
    agent.save_q_table(filename)
    print(f"\nTraining completed!")
    print(f"Final win rate: {wins / episodes * 100:.1f}%")
    print(f"Final Q-table size: {len(agent.q_table)}")


if __name__ == "__main__":
    # Train tiger agent
    print("=" * 50)
    print("Training Tiger Agent")
    print("=" * 50)
    tiger_agent = QLearningAgent(
        player='tiger',
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    train_agent(tiger_agent, episodes=1000, opponent='random')
    
    # Train goat agent
    print("\n" + "=" * 50)
    print("Training Goat Agent")
    print("=" * 50)
    goat_agent = QLearningAgent(
        player='goat',
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    train_agent(goat_agent, episodes=1000, opponent='random')

