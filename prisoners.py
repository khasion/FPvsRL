import numpy as np

# Define the payoff matrix for the Prisoner’s Dilemma
PAYOFF_MATRIX = np.array([
    [-1, -3],  # Cooperate
    [0, -2]    # Defect
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
n_actions = 2  # Cooperate, Defect

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
    reward_ql = PAYOFF_MATRIX[ql_action, fp_action]  # Payoff for Q-Learning agent

    # Update agents
    fp_agent.update(ql_action)
    ql_agent.update(ql_action, reward_ql)

    # Track outcomes
    if reward_fp > reward_ql:
        fp_wins += 1
    elif reward_fp < reward_ql:
        ql_wins += 1
    else:
        draws += 1

# Results
print("Results after", episodes, "episodes:") 
print(f"Fictitious Play wins: {fp_wins}")
print(f"Q-Learning wins: {ql_wins}")
print(f"Draws: {draws}")
