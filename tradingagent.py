import numpy as np
import pandas as pd
import random
from collections import defaultdict
import pickle
import sys
import os

# Add numpy compatibility for pickle loading
# This helps resolve the "ModuleNotFoundError: No module named 'numpy._core.numeric'" error
if not hasattr(np, "_core"):
    np._core = np.core
if not hasattr(np._core, "numeric"):
    np._core.numeric = np.core.numeric


class TradingAgent:
    def __init__(self):
        # Initialize Q-table as a nested defaultdict
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.alpha = 0.2  # Learning rate
        self.gamma = 0.95  # Discount factor
        self.epsilon = 0.3  # Exploration rate
        self.epsilon_decay = 0.99  # Decay rate for exploration
        self.min_epsilon = 0.05  # Minimum exploration rate

    def reward_function(self, history):
        # Sharpe-ratio inspired reward that penalizes volatility
        returns = np.log(history["portfolio_valuation", -1] / history["portfolio_valuation", -2])
        
        # Add penalty for large position changes to discourage excessive trading
        position_change = abs(history["position", -1] - history["position", -2])
        trading_penalty = 0.01 * position_change
        
        return returns - trading_penalty

    def make_features(self, df):
        # Price-based features
        df["feature_close"] = df["close"].pct_change()
        df["feature_open"] = df["close"] / df["open"]
        df["feature_high"] = df["high"] / df["close"]
        df["feature_low"] = df["low"] / df["close"]
        
        # Volume features
        df["feature_volume"] = df["volume"] / df["volume"].rolling(7 * 24).max()
        
        # Add moving averages
        df["sma_5"] = df["close"].rolling(window=5).mean() / df["close"]
        df["sma_20"] = df["close"].rolling(window=20).mean() / df["close"]
        df["sma_ratio"] = df["sma_5"] / df["sma_20"]
        
        # Add Bollinger Bands (20-period, 2 standard deviations)
        rolling_std = df["close"].rolling(window=20).std()
        df["bb_upper"] = (df["close"].rolling(window=20).mean() + 2 * rolling_std) / df["close"]
        df["bb_lower"] = (df["close"].rolling(window=20).mean() - 2 * rolling_std) / df["close"]
        df["bb_width"] = (df["bb_upper"] * df["close"] - df["bb_lower"] * df["close"]) / df["close"]
        
        # Add RSI (14-period)
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))
        df["rsi"] = df["rsi"] / 100  # Normalize to 0-1
        
        # Momentum
        df["momentum_5"] = df["close"].pct_change(periods=5)
        
        df.dropna(inplace=True)
        return df

    def get_position_list(self):
        return [x / 10.0 for x in range(-10, 21)]

    def _discretize_state(self, observation):
        """Convert continuous observation to discrete state for Q-table"""
        features = []
        
        # Discretize each feature in observation
        for i, value in enumerate(observation):
            if i == 0:  # Position feature (already discrete)
                features.append(int(value * 10))
            else:
                # Discretize other features into 10 bins
                if value < -0.05:
                    features.append(0)
                elif value < -0.02:
                    features.append(1)
                elif value < -0.01:
                    features.append(2)
                elif value < 0:
                    features.append(3)
                elif value < 0.01:
                    features.append(4)
                elif value < 0.02:
                    features.append(5)
                elif value < 0.05:
                    features.append(6)
                elif value < 0.1:
                    features.append(7)
                elif value < 0.2:
                    features.append(8)
                else:
                    features.append(9)
        
        # Return tuple for hashable state
        return tuple(features)

    def _choose_action(self, state, training=True):
        """Choose action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            # Explore: choose random action
            return random.randint(0, len(self.get_position_list()) - 1)
        else:
            # Exploit: choose best action from Q-table
            state_actions = self.q_table[state]
            if not state_actions:
                # If state not in Q-table, choose random action
                return random.randint(0, len(self.get_position_list()) - 1)
            
            # Find action with maximum Q-value
            return max(state_actions.items(), key=lambda x: x[1])[0]

    def train(self, env):
        num_episodes = 50  # Train over multiple episodes
        
        for episode in range(num_episodes):
            try:
                done, truncated = False, False
                observation, info = env.reset()
                
                state = self._discretize_state(observation)
                total_reward = 0
                
                while not done and not truncated:
                    # Choose action using epsilon-greedy
                    action = self._choose_action(state, training=True)
                    
                    # Take action and observe next state and reward
                    next_observation, reward, done, truncated, info = env.step(action)
                    next_state = self._discretize_state(next_observation)
                    total_reward += reward
                    
                    # Get maximum Q-value for next state
                    next_max_q = max([self.q_table[next_state][a] for a in range(len(self.get_position_list()))], default=0)
                    
                    # Update Q-value using Q-learning update rule
                    self.q_table[state][action] = (1 - self.alpha) * self.q_table[state][action] + \
                                                self.alpha * (reward + self.gamma * next_max_q)
                    
                    # Move to next state
                    state = next_state
                
                # Decay exploration rate
                self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
                
                # Print progress
                if (episode + 1) % 5 == 0:
                    print(f"Episode {episode + 1}/{num_episodes}, Total reward: {total_reward:.2f}, Epsilon: {self.epsilon:.3f}")
            except Exception as e:
                print(f"Error in episode {episode}: {str(e)}")
                continue

    def get_test_position(self, observation):
        """Use learned policy for testing"""
        state = self._discretize_state(observation)
        action = self._choose_action(state, training=False)
        return action

    def test(self, env, n_epochs):
        # DO NOT CHANGE - all changes will be ignored after upload to BRUTE!
        for _ in range(n_epochs):
            done, truncated = False, False
            observation, info = env.reset()
            while not done and not truncated:
                new_position = self.get_test_position(observation)
                observation, reward, done, truncated, info = env.step(new_position)
