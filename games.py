import numpy as np

class MatchingPenniesEnv:
    def __init__(self):
        self.actions = [0, 1]  # 0 = Heads, 1 = Tails
    
    def get_payoffs(self, action1, action2):
        if action1 == action2:
            return (1, -1)  # Player 1 wins
        else:
            return (-1, 1)  # Player 2 wins

class SecurityGameEnv:
    def __init__(self):
        self.states = [0, 1]  # 0 = Room A, 1 = Room B
        self.actions = [0, 1]  # 0 = Guard/Attack A, 1 = Guard/Attack B
        self.current_state = np.random.choice(self.states)
        self.transition_prob = 0.2  # Probability to reset state
    
    def get_payoffs(self, action_defender, action_attacker):
        if action_defender == action_attacker:
            return (-1, 1)  # Defender successfully guards
        else:
            return (1, -1)  # Attacker succeeds
    
    def step(self):
        # Transition to new state with 20% probability
        if np.random.rand() < self.transition_prob:
            self.current_state = np.random.choice(self.states)
        return self.current_state
    
class FPAgent:
    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.opponent_counts = np.ones(n_actions)  # Laplace smoothing
    
    def update_beliefs(self, opponent_action):
        self.opponent_counts[opponent_action] += 1
    
    def get_empirical_distribution(self):
        return self.opponent_counts / np.sum(self.opponent_counts)
    
    def choose_action(self, payoff_matrix):
        empirical_probs = self.get_empirical_distribution()
        expected_payoffs = payoff_matrix @ empirical_probs
        return np.argmax(expected_payoffs)  # Best response

class RLAgent:
    def __init__(self, n_actions, n_states=1, lr=0.1, gamma=0.95):
        self.n_actions = n_actions
        self.n_states = n_states
        self.lr = lr
        self.gamma = gamma
        self.Q = np.zeros((n_states, n_actions, n_actions))  # Q[state][a1][a2]
    
    def update(self, state, own_action, opp_action, reward, next_state):
        # Minimax-Q update: max_a1 min_a2 Q[s][a1][a2]
        future_value = np.max(np.min(self.Q[next_state], axis=1))
        self.Q[state][own_action][opp_action] += self.lr * (
            reward + self.gamma * future_value - self.Q[state][own_action][opp_action]
        )
    
    def choose_action(self, state, epsilon=0.1):
        if np.random.rand() < epsilon:
            return np.random.choice(self.n_actions)
        else:
            # Maximin strategy: max over min Q-values
            return np.argmax(np.min(self.Q[state], axis=1))

def run_matching_pennies_experiment(n_episodes=1000):
    env = MatchingPenniesEnv()
    player1 = FPAgent(n_actions=2)
    player2 = RLAgent(n_actions=2)
    
    # Payoff matrix for Player 1 (since it's zero-sum, Player 2's matrix is -1 * this)
    payoff_matrix = np.array([[1, -1], [-1, 1]])
    
    rewards_p1, rewards_p2 = [], []
    empirical_probs_history = []
    
    for _ in range(n_episodes):
        # Agents choose actions
        action1 = player1.choose_action(payoff_matrix)
        action2 = player2.choose_action(state=0)  # Single-state game
        
        # Get payoffs
        payoff1, payoff2 = env.get_payoffs(action1, action2)
        
        # Update beliefs/Q-values
        player1.update_beliefs(action2)
        player2.update(0, action2, action1, payoff2, 0)  # State=0
        
        # Log data
        rewards_p1.append(payoff1)
        rewards_p2.append(payoff2)
        empirical_probs_history.append(player1.get_empirical_distribution())
    
    return rewards_p1, rewards_p2, empirical_probs_history

def run_security_game_experiment(n_episodes=1000):
    env = SecurityGameEnv()
    defender = FPAgent(n_actions=2)
    attacker = RLAgent(n_actions=2, n_states=2)
    
    rewards_defender, rewards_attacker = [], []
    
    for _ in range(n_episodes):
        state = env.current_state
        action_defender = defender.choose_action(payoff_matrix=np.array([[-1, 1], [1, -1]]))
        action_attacker = attacker.choose_action(state)
        
        payoff_defender, payoff_attacker = env.get_payoffs(action_defender, action_attacker)
        next_state = env.step()
        
        # Update agents
        defender.update_beliefs(action_attacker)
        attacker.update(state, action_attacker, action_defender, payoff_attacker, next_state)
        
        # Log data
        rewards_defender.append(payoff_defender)
        rewards_attacker.append(payoff_attacker)
    
    return rewards_defender, rewards_attacker

import matplotlib.pyplot as plt

# Run Matching Pennies Experiment
mp_p1_rewards, mp_p2_rewards, emp_probs = run_matching_pennies_experiment()

# Plot convergence in Matching Pennies
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(np.array(emp_probs)[:, 0], label='Prob(Opponent Plays H)')
plt.axhline(0.5, linestyle='--', color='k', label='Nash Equilibrium')
plt.xlabel('Episode')
plt.ylabel('Probability')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(np.cumsum(mp_p1_rewards), label='FP Agent (Player 1)')
plt.plot(np.cumsum(mp_p2_rewards), label='RL Agent (Player 2)')
plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
plt.legend()
plt.tight_layout()
plt.show()

# Run Security Game Experiment
sg_def_rewards, sg_atk_rewards = run_security_game_experiment()

# Plot Security Game Results
plt.figure()
plt.plot(np.cumsum(sg_def_rewards), label='Defender (FP)')
plt.plot(np.cumsum(sg_atk_rewards), label='Attacker (RL)')
plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
plt.legend()
plt.show()
