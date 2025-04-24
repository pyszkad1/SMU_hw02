import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque, namedtuple

# Q-Network definition
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 516)
        self.fc3 = nn.Linear(516, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Replay buffer for experience replay
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        return Transition(*zip(*transitions))

    def __len__(self):
        return len(self.memory)

class TradingAgent:
    def __init__(self,
                 state_size=7,
                 action_size=31,
                 lr=1e-3,
                 gamma=0.99,
                 batch_size=64,
                 memory_size=10000,
                 epsilon_start=0.5,
                 epsilon_min=0.05,
                 epsilon_decay=0.95,
                 target_update_every=1000,
                 train_episodes=30):
        # Environment dimensions
        self.state_size = state_size
        self.action_size = action_size

        # Hyperparameters
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.train_episodes = train_episodes
        self.target_update_every = target_update_every

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.model = QNetwork(state_size, action_size).to(self.device)
        self.target_model = QNetwork(state_size, action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Replay buffer
        self.memory = ReplayBuffer(memory_size)

    def reward_function(self, history):
        # Default reward: log return of portfolio valuation
        return np.log(history["portfolio_valuation", -1] / history["portfolio_valuation", -2])

    def make_features(self, df):
        df["feature_close"] = df["close"].pct_change()
        df["feature_open"]  = df["close"] / df["open"]
        df["feature_high"]  = df["high"]  / df["close"]
        df["feature_low"]   = df["low"]   / df["close"]
        df["feature_volume"] = df["volume"] / df["volume"].rolling(7*24).max()
        df.dropna(inplace=True)
        return df

    def get_position_list(self):
        return [x / 10.0 for x in range(-10, 21)]

    def train(self, env):
        total_steps = 0
        for ep in range(self.train_episodes):
            try:
                print(f"Training episode {ep+1}/{self.train_episodes}")
                done = False
                truncated = False
                obs, info = env.reset()
                state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                total_reward = 0
                for i in range(5000):
                    if done or truncated:
                        break
                    if i % 1000 == 0:
                        print(f"Step {i} of episode {ep+1}")
                    # Epsilon-greedy action selection
                    if random.random() < self.epsilon:
                        action = random.randrange(self.action_size)
                    else:
                        with torch.no_grad():
                            q_values = self.model(state)
                            action = q_values.argmax().item()

                    next_obs, reward, done, truncated, info = env.step(action)
                    next_state = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                    total_reward += reward
                    # Store transition
                    self.memory.push(state, action, reward, next_state, done)
                    state = next_state
                    total_steps += 1

                    # Learn
                    if len(self.memory) >= self.batch_size:
                        self._optimize()

                    # Update target network
                    if total_steps % self.target_update_every == 0:
                        self.target_model.load_state_dict(self.model.state_dict())

                # Decay epsilon
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
                print(f"Episode {ep + 1}/{self.train_episodes}, Total reward: {total_reward:.2f}, Epsilon: {self.epsilon:.3f}")

            except Exception as e:
                print(f"Error in episode {ep+1}: {str(e)}")
                continue

    def _optimize(self):
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*transitions)

        state_batch = torch.cat(batch.state)
        action_batch = torch.tensor(batch.action, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_state_batch = torch.cat(batch.next_state)
        done_mask = torch.tensor(batch.done, dtype=torch.bool, device=self.device)

        # Current Q values
        current_q = self.model(state_batch).gather(1, action_batch)

        # Compute target Q values
        next_q = torch.zeros(self.batch_size, 1, device=self.device)
        with torch.no_grad():
            next_q_vals = self.target_model(next_state_batch).max(1)[0].unsqueeze(1)
        next_q[~done_mask] = next_q_vals[~done_mask]
        expected_q = reward_batch + (self.gamma * next_q)

        # Loss
        loss = F.mse_loss(current_q, expected_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_test_position(self, observation):
        # Use the learned policy: choose action with highest Q-value
        state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return q_values.argmax().item()

    def test(self, env, n_epochs):
        for _ in range(n_epochs):
            done = False
            truncated = False
            observation, info = env.reset()
            while not done and not truncated:
                action = self.get_test_position(observation)
                observation, reward, done, truncated, info = env.step(action)
