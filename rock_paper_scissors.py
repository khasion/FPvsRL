import numpy as np
import matplotlib.pyplot as plt

# Define the payoff matrix for Rock-Paper-Scissors
PAYOFF_MATRIX = np.array([
    [0, -1, 1],  # Rock
    [1, 0, -1],  # Paper
    [-1, 1, 0]   # Scissors
])

class FictitiousPlay:
    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.strategy = np.ones(n_actions) / n_actions  # Start with uniform strategy
        self.history = np.zeros(n_actions)  # Track opponent actions

    def update(self, opponent_action):
        self.history[opponent_action] += 1
        self.strategy = self.history / np.sum(self.history)

    def choose_action(self):
        return np.random.choice(range(self.n_actions), p=self.strategy)

class QLearning:
    def __init__(self, n_actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.n_actions = n_actions
        self.q_table = np.zeros(n_actions)
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate

    def choose_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(range(self.n_actions))  # Explore
        return np.argmax(self.q_table)  # Exploit

    def update(self, action, reward):
        best_next_q = np.max(self.q_table)
        self.q_table[action] += self.alpha * (reward + self.gamma * best_next_q - self.q_table[action])

# Simulation parameters
episodes = 1000
n_actions = 3  # Rock, Paper, Scissors

# Initialize agents
fp_agent = FictitiousPlay(n_actions)
ql_agent = QLearning(n_actions)

# Tracking results
fp_wins, ql_wins, draws = 0, 0, 0

for episode in range(episodes):
    # Agents choose actions
    fp_action = fp_agent.choose_action()
    ql_action = ql_agent.choose_action()

    # Determine rewards
    reward_fp = PAYOFF_MATRIX[fp_action, ql_action]
    reward_ql = -reward_fp  # Zero-sum

    # Update agents
    fp_agent.update(ql_action)
    ql_agent.update(ql_action, reward_ql)

    # Track outcomes
    if reward_fp > 0:
        fp_wins += 1
    elif reward_fp < 0:
        ql_wins += 1
    else:
        draws += 1

# Results
print("Results after", episodes, "episodes:")
print(f"Fictitious Play wins: {fp_wins}")
print(f"Q-Learning wins: {ql_wins}")
print(f"Draws: {draws}")

# Data For Prisoners Dilemma
results = {
    "Fictitious Play Wins": fp_wins,
    "Q-Learning Wins": ql_wins,
    "Draws": draws
}

# Output Figures
plt.figure(figsize=(8, 5))
plt.bar(list(results.keys()), list(results.values()), color=['blue', 'green', 'gray'])
plt.title("Results for Rock-Paper-Scissors")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("rps_results.png")
plt.close()

print("Figures generated and saved as 'rps_results.png'.")