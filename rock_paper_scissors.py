# rock_paper_scissors.py
import numpy as np
import matplotlib.pyplot as plt

# Stochastic Payoff Matrix with noise
BASE_PAYOFF = np.array([
    [0, -1, 1],  # Rock
    [1, 0, -1],  # Paper
    [-1, 1, 0]   # Scissors
])

def stochastic_payoff():
    noise = np.random.normal(0, 0.1, BASE_PAYOFF.shape)
    return BASE_PAYOFF + noise

class FictitiousPlay:
    def __init__(self, n_actions):
        self.n_actions = n_actions # 0: Rock, 1: Paper, 2: Scissors
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
    def __init__(self, n_actions, alpha=0.1, gamma=0.9, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.9995):
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
n_actions = 3
trials = 5

# Data collection
all_fp_rewards = np.zeros((trials, episodes))
all_ql_rewards = np.zeros((trials, episodes))
cumulative_scores_rps = np.zeros((trials, episodes, 2))  # FP, QL
all_ql_action_histories = []
all_fp_action_histories = []

for trial in range(trials):
    fp_agent = FictitiousPlay(n_actions)
    ql_agent = QLearning(n_actions)
    
    fp_cumulative = 0
    ql_cumulative = 0

    for episode in range(episodes):
        fp_action = fp_agent.choose_action()
        ql_action = ql_agent.choose_action()
        
        PAYOFF = stochastic_payoff()
        reward_fp = PAYOFF[fp_action, ql_action]
        reward_ql = -reward_fp
        
        fp_agent.update(ql_action)
        ql_agent.update(ql_action, reward_ql)
        
        fp_cumulative += reward_fp
        ql_cumulative += reward_ql

        all_fp_rewards[trial, episode] = reward_fp
        all_ql_rewards[trial, episode] = reward_ql
        cumulative_scores_rps[trial, episode] = [fp_cumulative, ql_cumulative]
    all_ql_action_histories.append(ql_agent.action_history)
    all_fp_action_histories.append(fp_agent.action_history)

# Plotting
def moving_average(x, window=100):
    return np.convolve(x, np.ones(window)/window, mode='valid')

# QL Strategy Evolution
window_size = 100
ql_action_freq = np.zeros((3, episodes))
fp_action_freq = np.zeros((3, episodes))
for action in range(3):
    for ep in range(episodes):
        # Calculate frequency across all trials
        count = 0
        for trial_hist in all_ql_action_histories:
            count += trial_hist[:ep+1].count(action)
        ql_action_freq[action, ep] = count / (trials * (ep+1))

        count = 0
        for trial_hist in all_fp_action_histories:
            count += trial_hist[:ep+1].count(action)
        fp_action_freq[action, ep] = count / (trials * (ep+1))
        

plt.figure(figsize=(12,6))
for i, label in enumerate(['Rock', 'Paper', 'Scissors']):
    plt.plot(ql_action_freq[i], label=label)
plt.title("QL Action Selection Frequency (RPS)")
plt.xlabel("Episode")
plt.ylabel("Frequency")
plt.legend()
plt.savefig("rps_ql_strategy.png")
plt.close()

plt.figure(figsize=(12,6))
for i, label in enumerate(['Rock', 'Paper', 'Scissors']):
    plt.plot(fp_action_freq[i], label=label)
plt.title("FP Action Selection Frequency (RPS)")
plt.xlabel("Episode")
plt.ylabel("Frequency")
plt.legend()
plt.savefig("rps_fp_strategy.png")
plt.close()

# Cumulative Scores
plt.figure(figsize=(12,6))
plt.plot(np.mean(cumulative_scores_rps[:,:,0], axis=0), label='FP Cumulative')
plt.plot(np.mean(cumulative_scores_rps[:,:,1], axis=0), label='QL Cumulative')
plt.title("Cumulative Scores (RPS)")
plt.xlabel("Episode")
plt.ylabel("Score")
plt.legend()
plt.savefig("rps_cumulative.png")
plt.close()

# Behavior Comparison
plt.figure(figsize=(12,6))
plt.plot(moving_average(np.mean(all_fp_rewards, axis=0)), label='FP')
plt.plot(moving_average(np.mean(all_ql_rewards, axis=0)), label='QL')
plt.title("Moving Average Reward Comparison (RPS)")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
plt.savefig("rps_rewards_comparison.png")
plt.close()