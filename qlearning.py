'''
Q-Learning Agent for Huligutta (Goats and Tigers)
file: qlearning.py
Description: Q-learning implementation for the game
'''

__author__ = "Clyde James Felix"
__email__ = "cjfelix.hawaii.edu"
__status__ = "Dev"

import numpy as np
import pickle
import random
from huligutta import *
from functions import *
from copy import deepcopy
import os

class QLearningAgent:
    """
    Q-Learning agent for playing Huligutta game.
    Can play as either Tigers or Goats.
    """
    
    def __init__(self, player='tiger', learning_rate=0.1, discount_factor=0.9, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        """
        Initialize Q-learning agent.
        
        Args:
            player: 'tiger' or 'goat'
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon: Initial exploration rate
            epsilon_decay: Rate at which epsilon decays
            epsilon_min: Minimum epsilon value
        """
        self.player = player
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table: state -> action -> Q-value
        # State is represented as a tuple of board positions
        self.q_table = {}
        
        # Training statistics
        self.training_stats = {
            'episodes': 0,
            'wins': 0,
            'losses': 0,
            'draws': 0
        }
        
    def get_state_key(self, board_positions):
        """
        Convert board state to a hashable key for Q-table.
        
        Args:
            board_positions: Dictionary of board positions
            
        Returns:
            Tuple representing the state
        """
        # Create a sorted tuple of (position, content) pairs
        state_items = []
        for pos in sorted(board_positions.keys()):
            content = board_positions[pos]
            if content == 'X':
                state_items.append((pos, 'X'))
            elif content == 'O':
                state_items.append((pos, 'O'))
            else:
                state_items.append((pos, 'E'))  # Empty
        
        return tuple(state_items)
    
    def get_state_key_simple(self, board_positions):
        """
        Simplified state representation using only piece positions.
        More efficient for Q-table lookup.
        """
        tigers = tuple(sorted(tigerPositions(board_positions)))
        goats = tuple(sorted(goatPositions(board_positions)))
        return (tigers, goats)
    
    def get_actions(self, board_positions, is_tiger_turn):
        """
        Get all possible actions for the current player.
        
        Args:
            board_positions: Current board state
            is_tiger_turn: True if it's tiger's turn, False for goat's turn
            
        Returns:
            List of actions, where each action is (from_pos, to_pos) or (None, to_pos) for placement
        """
        actions = []
        
        if is_tiger_turn:
            # Tiger actions
            tigers = tigerPositions(board_positions)
            for tiger_pos in tigers:
                tiger = Tiger(tiger_pos)
                possible_moves = tiger.possibleMoves()
                captures = Position(tiger_pos[0], tiger_pos[1]).get_captures()
                captures_list = captures if captures is not None else []
                for move_pos in possible_moves:
                    # Check if it's a capture move
                    if move_pos in captures_list:
                        actions.append(('capture', tiger_pos, move_pos))
                    else:
                        actions.append(('move', tiger_pos, move_pos))
        else:
            # Goat actions
            goats = goatPositions(board_positions)
            empty = emptyPositions(board_positions)
            goat_count = len(goats)
            
            if goat_count < 15:
                # Placement phase
                for empty_pos in empty:
                    actions.append(('place', None, empty_pos))
            else:
                # Movement phase
                for goat_pos in goats:
                    goat = Goat(goat_pos)
                    neighbors = Position(goat_pos[0], goat_pos[1]).get_neighbors()
                    for neighbor in neighbors:
                        if Position(neighbor[0], neighbor[1]).content() == ():
                            actions.append(('move', goat_pos, neighbor))
        
        return actions
    
    def choose_action(self, board_positions, is_tiger_turn, training=True):
        """
        Choose an action using epsilon-greedy policy.
        
        Args:
            board_positions: Current board state
            is_tiger_turn: True if it's tiger's turn
            training: Whether we're in training mode (affects exploration)
            
        Returns:
            Selected action: (action_type, from_pos, to_pos)
        """
        state_key = self.get_state_key_simple(board_positions)
        actions = self.get_actions(board_positions, is_tiger_turn)
        
        if not actions:
            return None
        
        # Epsilon-greedy action selection
        if training and random.random() < self.epsilon:
            # Explore: random action
            return random.choice(actions)
        else:
            # Exploit: choose best action based on Q-values
            best_action = None
            best_q_value = float('-inf') if is_tiger_turn else float('-inf')
            
            # Initialize Q-values for unseen state-action pairs
            if state_key not in self.q_table:
                self.q_table[state_key] = {}
            
            for action in actions:
                action_key = self._action_to_key(action)
                if action_key not in self.q_table[state_key]:
                    self.q_table[state_key][action_key] = 0.0
                
                q_value = self.q_table[state_key][action_key]
                
                if q_value > best_q_value:
                    best_q_value = q_value
                    best_action = action
            
            # If all Q-values are the same, choose randomly
            if best_action is None:
                return random.choice(actions)
            
            return best_action
    
    def _action_to_key(self, action):
        """Convert action tuple to hashable key."""
        return action
    
    def update_q_value(self, state, action, reward, next_state, done):
        """
        Update Q-value using Q-learning formula.
        
        Q(s,a) = Q(s,a) + alpha * [r + gamma * max(Q(s',a')) - Q(s,a)]
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state after action
            done: Whether episode is finished
        """
        state_key = self.get_state_key_simple(state)
        action_key = self._action_to_key(action)
        
        # Initialize Q-values if needed
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        if action_key not in self.q_table[state_key]:
            self.q_table[state_key][action_key] = 0.0
        
        # Calculate max Q-value for next state
        if done:
            max_next_q = 0.0
        else:
            next_state_key = self.get_state_key_simple(next_state)
            if next_state_key in self.q_table and len(self.q_table[next_state_key]) > 0:
                max_next_q = max(self.q_table[next_state_key].values())
            else:
                max_next_q = 0.0
        
        # Q-learning update
        current_q = self.q_table[state_key][action_key]
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state_key][action_key] = new_q
    
    def get_reward(self, board_positions, is_tiger_turn, game_over, winner):
        """
        Calculate reward for the current state.
        
        Args:
            board_positions: Current board state
            is_tiger_turn: Whether it's the agent's turn
            game_over: Whether game is finished
            winner: 'tiger', 'goat', or None
            
        Returns:
            Reward value
        """
        if not game_over:
            # Small reward for making progress
            # Tigers get reward for capturing goats
            if self.player == 'tiger':
                goats_eaten = 5 - len(goatPositions(board_positions))
                return goats_eaten * 0.1
            else:
                # Goats get reward for blocking tigers
                tigers = tigerPositions(board_positions)
                total_moves = sum(len(Tiger(t).possibleMoves()) for t in tigers)
                return (10 - total_moves) * 0.1
        
        # Game over rewards
        if winner == self.player:
            return 100.0  # Win
        elif winner is not None:
            return -100.0  # Loss
        else:
            return 0.0  # Draw (shouldn't happen in this game)
    
    def decay_epsilon(self):
        """Decay epsilon for exploration."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_q_table(self, filepath='q_table.pkl'):
        """Save Q-table to file."""
        save_data = {
            'q_table': self.q_table,
            'epsilon': self.epsilon,
            'training_stats': self.training_stats,
            'player': self.player
        }
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"Q-table saved to {filepath}")
    
    def load_q_table(self, filepath='q_table.pkl'):
        """Load Q-table from file."""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                save_data = pickle.load(f)
                self.q_table = save_data.get('q_table', {})
                self.epsilon = save_data.get('epsilon', self.epsilon_min)
                self.training_stats = save_data.get('training_stats', self.training_stats)
                self.player = save_data.get('player', self.player)
            print(f"Q-table loaded from {filepath}")
            return True
        else:
            print(f"Q-table file {filepath} not found")
            return False
    
    def get_stats(self):
        """Get training statistics."""
        return self.training_stats.copy()

