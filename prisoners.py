# prisoners.py
import numpy as np
import matplotlib.pyplot as plt

# Zero-sum payoff matrix (Row: Cooperate/Defect, Column: Cooperate/Defect)
PAYOFF_MATRIX_ROW = np.array([
    [1, -3],   # Cooperate
    [3, -1]    # Defect
])  # Row player's payoffs (Column player gets negative)

def get_payoff(action_row, action_col):
    base = PAYOFF_MATRIX_ROW[action_row, action_col]
    return base + np.random.normal(0, 0.5)  # Add noise

class FictitiousPlay:
    def __init__(self, n_actions):
        self.n_actions = n_actions # 0: Cooperate, 1: Defect
        self.strategy = np.ones(n_actions) / n_actions
        self.history = np.zeros(n_actions)
        self.strategy_history = []
        self.action_history = []

    def update(self, opponent_action):
        self.history[opponent_action] += 1
        self.strategy = self.history / np.sum(self.history)
        self.strategy_history.append(self.strategy.copy())

    def choose_action(self):
        # Generates a random sample
        # a : a random sample is generated from its elements.
        # p : The probabilities associated with each entry in a.
        action = np.random.choice(a = range(self.n_actions), p = self.strategy)
        self.action_history.append(action)
        return action

class QLearning:
    def __init__(self, n_actions, alpha=0.1, gamma=0.9, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.999):
        self.n_actions = n_actions
        self.q_table = np.zeros(n_actions)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.q_history = []
        self.action_history = []

    def choose_action(self):
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.n_actions)
        else:
            action = np.argmax(self.q_table)
        self.action_history.append(action)
        return action

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon*self.epsilon_decay)

    def update(self, action, reward):
        best_next_q = np.max(self.q_table)
        self.q_table[action] += self.alpha * (reward + self.gamma * best_next_q - self.q_table[action])
        self.q_history.append(self.q_table.copy())
        self.update_epsilon()

# Simulation parameters
episodes = 2000
n_actions = 2
trials = 5

# Data collection
all_fp_rewards_pd = np.zeros((trials, episodes))
all_ql_rewards_pd = np.zeros((trials, episodes))

all_fp_defections = np.zeros((trials, episodes))
all_ql_defections = np.zeros((trials, episodes))

cumulative_scores_pd = np.zeros((trials, episodes, 2))  # FP, QL
all_ql_action_histories = []

for trial in range(trials):
    fp_agent = FictitiousPlay(n_actions)
    ql_agent = QLearning(n_actions, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.999)
    
    fp_cumulative = 0
    ql_cumulative = 0
    
    for episode in range(episodes):
        fp_action = fp_agent.choose_action()
        ql_action = ql_agent.choose_action()
        
        reward_fp = get_payoff(fp_action, ql_action)
        reward_ql = -reward_fp
        
        fp_agent.update(ql_action)
        ql_agent.update(ql_action, reward_ql)
        
        fp_cumulative += reward_fp
        ql_cumulative += reward_ql
        
        all_fp_defections[trial, episode] = fp_action
        all_ql_defections[trial, episode] = ql_action

        all_fp_rewards_pd[trial, episode] = reward_fp
        all_ql_rewards_pd[trial, episode] = reward_ql
        cumulative_scores_pd[trial, episode] = [fp_cumulative, ql_cumulative]
    all_ql_action_histories.append(ql_agent.action_history)

# Plotting
def moving_average(x, window=100):
    return np.convolve(x, np.ones(window)/window, mode='valid')

# QL Strategy Evolution
window_size = 100
ql_action_freq = np.zeros((3, episodes))
for action in range(3):
    for ep in range(episodes):
        # Calculate frequency across all trials
        count = 0
        for trial_hist in all_ql_action_histories:
            count += trial_hist[:ep+1].count(action)
        ql_action_freq[action, ep] = count / (trials * (ep+1))

# Moving Average Reward Comparison
plt.figure(figsize=(12,6))
plt.plot(moving_average(np.mean(all_fp_rewards_pd, axis=0)), label='FP')
plt.plot(moving_average(np.mean(all_ql_rewards_pd, axis=0)), label='QL')
plt.title("Moving Average Rewards Comparison (PD)")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
plt.savefig("pd_rewards.png")
plt.close()

# Cumulative Scores
plt.figure(figsize=(12,6))
plt.plot(np.mean(cumulative_scores_pd[:,:,0], axis=0), label='FP')
plt.plot(np.mean(cumulative_scores_pd[:,:,1], axis=0), label='QL')
plt.title("Cumulative Scores (PD)")
plt.xlabel("Episode")
plt.ylabel("Score")
plt.legend()
plt.savefig("pd_cumulative.png")
plt.close()

# Behavior
plt.figure(figsize=(12, 6))
plt.plot(moving_average(np.mean(all_fp_defections, axis=0)), label='FP Defect Probability')
plt.plot(moving_average(np.mean(all_ql_defections, axis=0)), label='QL Defect Probability')
plt.title("Defection Behavior Over Time")
plt.xlabel("Episode")
plt.ylabel("Defect Metric")
plt.legend()
plt.savefig("pd_behavior.png")
plt.close()