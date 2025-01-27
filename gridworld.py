import numpy as np
import matplotlib.pyplot as plt

# Define the Gridworld environment
GRID_SIZE = 3
REWARDS = {
    (0, 0): 10,  # Target cell
    (2, 2): -5   # Penalty cell
}
ACTIONS = ["up", "down", "left", "right"]
ACTION_EFFECTS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1)
}

class Gridworld:
    def __init__(self, grid_size, rewards):
        self.grid_size = grid_size
        self.rewards = rewards
        self.state = np.zeros((grid_size, grid_size))
        for (x, y), reward in rewards.items():
            self.state[x, y] = reward

    def get_reward(self, position):
        return self.rewards.get(tuple(position), 0)

    def is_valid_move(self, position):
        x, y = position
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

class FictitiousPlayAgent:
    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.strategy = np.ones(n_actions) / n_actions  # Uniform strategy
        self.history = np.zeros(n_actions)  # Track opponent actions

    def update(self, opponent_action):
        self.history[opponent_action] += 1
        self.strategy = self.history / np.sum(self.history)

    def choose_action(self):
        return np.random.choice(range(self.n_actions), p=self.strategy)

class QLearningAgent:
    def __init__(self, grid_size, n_actions, alpha=0.1, gamma=0.9, epsilon=1):
        self.n_actions = n_actions
        self.q_table = np.zeros((grid_size, grid_size, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, state):
        x, y = state
        if np.random.rand() < self.epsilon:
            self.epsilon = self.epsilon - 0.1
            return np.random.choice(range(self.n_actions))  # Explore
        return np.argmax(self.q_table[x, y])  # Exploit

    def update(self, state, action, reward, next_state):
        x, y = state
        nx, ny = next_state
        best_next_q = np.max(self.q_table[nx, ny])
        self.q_table[x, y, action] += self.alpha * (reward + self.gamma * best_next_q - self.q_table[x, y, action])

# Simulation parameters
episodes = 100000
n_actions = len(ACTIONS)

# Initialize the environment and agents
env = Gridworld(GRID_SIZE, REWARDS)
fp_agent = FictitiousPlayAgent(n_actions)
ql_agent = QLearningAgent(GRID_SIZE, n_actions)

# Initialize positions
fp_position = [0, 0]
ql_position = [2, 2]

# Track scores
fp_score, ql_score = 0, 0
fp_wins, ql_wins, ties = 0, 0, 0

for episode in range(episodes):
    # Agents choose actions
    fp_action = fp_agent.choose_action()
    ql_action = ql_agent.choose_action(ql_position)

    # Calculate new positions
    fp_new_position = [
        fp_position[0] + ACTION_EFFECTS[ACTIONS[fp_action]][0],
        fp_position[1] + ACTION_EFFECTS[ACTIONS[fp_action]][1]
    ]
    ql_new_position = [
        ql_position[0] + ACTION_EFFECTS[ACTIONS[ql_action]][0],
        ql_position[1] + ACTION_EFFECTS[ACTIONS[ql_action]][1]
    ]

    # Validate moves
    if env.is_valid_move(fp_new_position):
        fp_position = fp_new_position
    if env.is_valid_move(ql_new_position):
        ql_position = ql_new_position

    # Get rewards
    fp_reward = env.get_reward(fp_position)
    ql_reward = -fp_reward  # Zero-sum property

    # Update agents
    fp_agent.update(ql_action)
    ql_agent.update(ql_position, ql_action, ql_reward, fp_position)

    # Update scores
    fp_score += fp_reward
    ql_score += ql_reward

    if fp_reward > ql_reward:
        fp_wins += 1
    elif ql_reward > fp_reward:
        ql_wins += 1
    else:
        ties += 1

    

# Results
print("Results after", episodes, "episodes:")
print(f"Fictitious Play Agent Score: {fp_score}")
print(f"Q-Learning Agent Score: {ql_score}")
print(f"Fictitious Play wins: {fp_wins}")
print(f"Q-Learning wins: {ql_wins}")
print(f"Ties: {ties}")

# Visualization
labels = ["Fictitious Play", "Q-Learning"]
scores = [fp_score, ql_score]

plt.figure(figsize=(8, 5))
plt.bar(labels, scores, color=["blue", "green"])
plt.title("Scores in Gridworld")
plt.ylabel("Score")
plt.tight_layout()
plt.savefig("gridworld_results.png")
plt.close()

print("Results plotted and saved as 'gridworld_results.png'.")