#!/usr/bin/env python
"""
Comparison of All Pairwise Combinations of Agents:
    - Fictitious Play (FP)
    - Q-Learning (QL)
    - Minimax RL (MM)
    - Belief-Based (BP)

This script defines a stochastic Rock-Paper-Scissors game (with two states)
and four agent types. It then runs a series of experiments comparing every pair
of agents and produces a suite of plots for each comparison:
    a. Moving Average Rewards
    b. Cumulative Scores
    c. Environment State Evolution
    d. Joint Action Frequency (Heatmap)
    e. Reward Distribution (with error bars)

Additionally, when applicable:
    f. Policy Evolution (for agents that record a strategy; e.g. FP)
    g. Epsilon Decay (for agents using ε-greedy, e.g. QL and MM)
    h. Q-Value Evolution and Convergence (for agents with Q-tables)

Each experiment's plots are saved with filenames that include the experiment name.
Processed data (i.e. the exact data used to produce the plots) are exported as CSV files.
Make sure to install the required packages:
    pip install numpy matplotlib scipy seaborn tqdm
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from scipy.optimize import linprog
import seaborn as sns
import os

# Create output directory for plots
output_dir = "rps-plots"
os.makedirs(output_dir, exist_ok=True)

############################################
# Environment: Stochastic Rock-Paper-Scissors
############################################
class StochasticRPSGame:
    def __init__(self):
        # Two states for the game (0 and 1)
        self.states = [0, 1]
        self.current_state = 0
        # In state 0: standard RPS payoffs; in state 1: payoffs are "flipped"
        self.payoff_matrices = {
            0: np.array([[ 0, -1,  1],
                         [ 1,  0, -1],
                         [-1,  1,  0]]),
            1: np.array([[ 0,  1, -1],
                         [-1,  0,  1],
                         [ 1, -1,  0]])
        }
        # State transition: high probability to remain in the same state.
        self.state_transition = np.array([[0.8, 0.2],
                                          [0.2, 0.8]])
        
    def step(self, action1, action2):
        """
        Given actions (0: Rock, 1: Paper, 2: Scissors) for player1 and player2,
        return (reward for player1, reward for player2) and update the state.
        """
        payoff = self.payoff_matrices[self.current_state]
        reward1 = payoff[action1, action2]
        reward2 = -reward1  # zero-sum game
        next_state = np.random.choice(self.states, p=self.state_transition[self.current_state])
        self.current_state = next_state
        return reward1, reward2, next_state

############################################
# Agent Classes
############################################

# 1. Fictitious Play Agent (FP)
class FictitiousPlayAgent:
    def __init__(self, n_actions):
        self.n_actions = n_actions
        # Initialize with a uniform prior (ones)
        self.history = np.ones(n_actions)
        self.strategy = self.history / np.sum(self.history)
        self.strategy_history = []  # record the evolving strategy
        self.action_history = []
        
    def choose_action(self, state):
        # Ignores state; uses empirical strategy
        action = np.random.choice(range(self.n_actions), p=self.strategy)
        self.action_history.append(action)
        return action
    
    def update(self, state, my_action, opponent_action, reward, next_state):
        self.history[opponent_action] += 1
        self.strategy = self.history / np.sum(self.history)
        self.strategy_history.append(self.strategy.copy())

# 2. Q-Learning Agent (QL)
class QLearningAgent:
    def __init__(self, n_actions, n_states, alpha=0.1, gamma=0.9,
                 epsilon=1.0, epsilon_decay=0.9995396, epsilon_min=0.1):
        self.n_actions = n_actions
        self.n_states = n_states
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        # Q-table: for each state, a vector (one value per action)
        self.q_table = {state: np.zeros(n_actions) for state in range(n_states)}
        self.q_history = {state: [] for state in range(n_states)}
        self.action_history = []
        self.epsilon_history = []  # record epsilon over time
        
    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.n_actions)
        else:
            action = np.argmax(self.q_table[state])
        self.action_history.append(action)
        return action
    
    def update(self, state, my_action, opponent_action, reward, next_state):
        best_next_q = np.max(self.q_table[next_state])
        prev = self.q_table[state][my_action]
        self.q_table[state][my_action] += self.alpha * (reward + self.gamma * best_next_q - prev)
        self.q_history[state].append(self.q_table[state].copy())
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.epsilon_history.append(self.epsilon)

# 3. Minimax RL Agent (MM)
class MinimaxRLAgent:
    def __init__(self, n_actions, n_states, alpha=0.1, gamma=0.9,
                 epsilon=1.0, epsilon_decay=0.9995396, epsilon_min=0.1):
        self.n_actions = n_actions
        self.n_states = n_states
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        # Q-table: for each state, a matrix (our actions x opponent actions)
        self.q_table = {state: np.zeros((n_actions, n_actions)) for state in range(n_states)}
        self.q_history = {state: [] for state in range(n_states)}
        self.action_history = []
        self.epsilon_history = []
        
    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.n_actions)
        else:
            v, p = solve_minimax(self.q_table[state])
            action = np.random.choice(self.n_actions, p=p)
        self.action_history.append(action)
        return action
    
    def update(self, state, my_action, opponent_action, reward, next_state):
        v_next, _ = solve_minimax(self.q_table[next_state])
        self.q_table[state][my_action, opponent_action] += self.alpha * (reward + self.gamma * v_next - self.q_table[state][my_action, opponent_action])
        self.q_history[state].append(self.q_table[state].copy())
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.epsilon_history.append(self.epsilon)

# 4. Belief-Based Agent (BP)
class BeliefBasedAgent:
    def __init__(self, n_actions, payoff_matrices):
        self.n_actions = n_actions
        # Assume the agent knows the payoff matrices.
        self.payoff_matrices = payoff_matrices  # dict: state -> payoff matrix
        # Beliefs: counts for each state (start with ones)
        self.beliefs = {state: np.ones(n_actions) for state in payoff_matrices.keys()}
        self.belief_history = {state: [] for state in payoff_matrices.keys()}
        self.action_history = []
        
    def choose_action(self, state):
        belief = self.beliefs[state]
        p_opponent = belief / np.sum(belief)
        exp_payoffs = self.payoff_matrices[state].dot(p_opponent)
        best_actions = np.where(exp_payoffs == np.max(exp_payoffs))[0]
        action = np.random.choice(best_actions)
        self.action_history.append(action)
        return action
    
    def update(self, state, my_action, opponent_action, reward, next_state):
        self.beliefs[state][opponent_action] += 1
        self.belief_history[state].append(self.beliefs[state].copy())

############################################
# Helper: LP Solver for the Minimax Agent
############################################
def solve_minimax(Q_mat):
    """
    Given a Q-matrix (n_actions x n_actions) for a zero-sum stage game,
    solve for the minimax value and the optimal mixed strategy.
    
    The LP formulation is:
        maximize v
        subject to: for each opponent action j, sum_i Q(i,j)*p(i) >= v,
                    sum_i p(i) = 1, and p(i) >= 0.
    This is transformed into a standard minimization LP.
    """
    n = Q_mat.shape[0]
    c = [-1] + [0] * n  # objective: minimize -v
    A_ub = []
    b_ub = []
    for j in range(n):
        constraint = [1] + [-Q_mat[i, j] for i in range(n)]
        A_ub.append(constraint)
        b_ub.append(0)
    A_eq = [[0] + [1] * n]
    b_eq = [1]
    bounds = [(None, None)] + [(0, 1) for _ in range(n)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method='highs')
    if res.success:
        x = res.x
        v = x[0]
        p = np.maximum(x[1:], 0)
        if np.sum(p) > 0:
            p = p / np.sum(p)
        else:
            p = np.ones(n) / n
        return v, p
    else:
        return 0, np.ones(n) / n

############################################
# Simulation Function (with extra recording)
############################################
def run_simulation(agent1_class, agent2_class, agent1_params, agent2_params,
                   game_env_class, episodes=5000, trials=5):
    """
    Run a match between two agents over many episodes and trials.
    Records for each trial:
      - Episode rewards and cumulative scores
      - Environment state sequence
      - Joint actions
      - Extra info (e.g. strategy history, epsilon history)
      - Per-episode Q-values (if available)
    Returns a dictionary with the collected data.
    """
    rewards_agent1_trials = []
    rewards_agent2_trials = []
    cum_scores_agent1_trials = []
    cum_scores_agent2_trials = []
    states_trials = []
    joint_actions_trials = []
    extra_info_agent1_trials = []
    extra_info_agent2_trials = []
    q_values_agent1_trials = []
    q_values_agent2_trials = []
    q_history_agent1_trials = []
    q_history_agent2_trials = []

    for t in range(trials):
        env = game_env_class()
        agent1 = agent1_class(**agent1_params)
        agent2 = agent2_class(**agent2_params)
        
        rewards1 = []
        rewards2 = []
        cum_scores1 = []
        cum_scores2 = []
        states = []
        joint_actions = []
        q_values1 = []  # per episode q_values for agent1
        q_values2 = []  # per episode q_values for agent2
        
        extra1 = {
            'policy_history': agent1.strategy_history if hasattr(agent1, 'strategy_history') else None,
            'epsilon_history': agent1.epsilon_history if hasattr(agent1, 'epsilon_history') else None,
            'belief_history': agent1.belief_history if hasattr(agent1, 'belief_history') else None,
        }
        extra2 = {
            'policy_history': agent2.strategy_history if hasattr(agent2, 'strategy_history') else None,
            'epsilon_history': agent2.epsilon_history if hasattr(agent2, 'epsilon_history') else None,
            'belief_history': agent2.belief_history if hasattr(agent2, 'belief_history') else None,
        }
        
        cum1, cum2 = 0, 0
        current_state = env.current_state
        
        for ep in range(episodes):
            states.append(current_state)
            a1 = agent1.choose_action(current_state)
            a2 = agent2.choose_action(current_state)
            joint_actions.append((a1, a2))
            r1, r2, next_state = env.step(a1, a2)
            agent1.update(current_state, a1, a2, r1, next_state)
            agent2.update(current_state, a2, a1, r2, next_state)
            if hasattr(agent1, 'q_table'):
                q_values1.append(agent1.q_table[current_state].copy())
            else:
                q_values1.append(None)
            if hasattr(agent2, 'q_table'):
                q_values2.append(agent2.q_table[current_state].copy())
            else:
                q_values2.append(None)
            
            cum1 += r1
            cum2 += r2
            rewards1.append(r1)
            rewards2.append(r2)
            cum_scores1.append(cum1)
            cum_scores2.append(cum2)
            current_state = next_state
        
        rewards_agent1_trials.append(rewards1)
        rewards_agent2_trials.append(rewards2)
        cum_scores_agent1_trials.append(cum_scores1)
        cum_scores_agent2_trials.append(cum_scores2)
        states_trials.append(states)
        joint_actions_trials.append(joint_actions)
        extra_info_agent1_trials.append(extra1)
        extra_info_agent2_trials.append(extra2)
        q_values_agent1_trials.append(q_values1)
        q_values_agent2_trials.append(q_values2)
        
        if hasattr(agent1, 'q_history'):
            q_history_agent1_trials.append(agent1.q_history.get(0, []))
        if hasattr(agent2, 'q_history'):
            q_history_agent2_trials.append(agent2.q_history.get(0, []))
    
    results = {
        'rewards_agent1': np.array(rewards_agent1_trials),
        'rewards_agent2': np.array(rewards_agent2_trials),
        'cum_scores_agent1': np.array(cum_scores_agent1_trials),
        'cum_scores_agent2': np.array(cum_scores_agent2_trials),
        'states': np.array(states_trials),
        'joint_actions': joint_actions_trials,
        'extra_info_agent1': extra_info_agent1_trials,
        'extra_info_agent2': extra_info_agent2_trials,
        'q_values_agent1': q_values_agent1_trials,
        'q_values_agent2': q_values_agent2_trials,
    }
    if q_history_agent1_trials:
        results['q_history_agent1'] = q_history_agent1_trials
    if q_history_agent2_trials:
        results['q_history_agent2'] = q_history_agent2_trials
    return results

############################################
# Plotting Helpers
############################################
def moving_average(x, window=100):
    return np.convolve(x, np.ones(window)/window, mode='valid')

def pad_and_average(time_series_list):
    max_len = max(len(arr) for arr in time_series_list)
    padded = np.array([np.pad(arr, (0, max_len - len(arr)), mode='constant', constant_values=np.nan)
                       for arr in time_series_list])
    return np.nanmean(padded, axis=0)

############################################
# Processed Data Export Lists
############################################
export_moving_avg = []
export_cumulative = []
export_states = []
export_joint_actions = []
export_reward_distribution = []
export_policy_evolution = []
export_epsilon_decay = []
export_q_evolution = []
export_q_convergence = []

############################################
# Define Experiments (all pairwise combinations)
############################################
n_actions = 3
n_states = 2
episodes = 5000
trials = 5

fp_params = {'n_actions': n_actions}
ql_params = {'n_actions': n_actions, 'n_states': n_states,
             'alpha': 0.1, 'gamma': 0.9, 'epsilon': 1.0,
             'epsilon_decay': 0.9995396, 'epsilon_min': 0.1}
mm_params = {'n_actions': n_actions, 'n_states': n_states,
             'alpha': 0.1, 'gamma': 0.9, 'epsilon': 1.0,
             'epsilon_decay': 0.9995396, 'epsilon_min': 0.1}
dummy_env = StochasticRPSGame()
bp_params = {'n_actions': n_actions, 'payoff_matrices': dummy_env.payoff_matrices}

experiments = [
    {"name": "fp_vs_ql", "agent1": FictitiousPlayAgent, "agent2": QLearningAgent,
     "params1": fp_params, "params2": ql_params, "labels": ("Fictitious Play", "Q-Learning")},
    {"name": "fp_vs_bp", "agent1": FictitiousPlayAgent, "agent2": BeliefBasedAgent,
     "params1": fp_params, "params2": bp_params, "labels": ("Fictitious Play", "Belief-Based")},
    {"name": "ql_vs_mm", "agent1": QLearningAgent, "agent2": MinimaxRLAgent,
     "params1": ql_params, "params2": mm_params, "labels": ("Q-Learning", "Minimax RL")},
    {"name": "fp_vs_mm", "agent1": FictitiousPlayAgent, "agent2": MinimaxRLAgent,
     "params1": fp_params, "params2": mm_params, "labels": ("Fictitious Play", "Minimax RL")},
    {"name": "ql_vs_bp", "agent1": QLearningAgent, "agent2": BeliefBasedAgent,
     "params1": ql_params, "params2": bp_params, "labels": ("Q-Learning", "Belief-Based")},
    {"name": "mm_vs_bp", "agent1": MinimaxRLAgent, "agent2": BeliefBasedAgent,
     "params1": mm_params, "params2": bp_params, "labels": ("Minimax RL", "Belief-Based")},
]

total_wins = {"Fictitious Play": 0, "Q-Learning": 0, "Minimax RL": 0, "Belief-Based": 0}

############################################
# Loop over Experiments: Generate Plots and Collect Processed Data
############################################
for exp in tqdm(experiments, desc="Processing Experiments"):
    exp_name = exp["name"]
    label1, label2 = exp["labels"]
    
    print(f"Running experiment: {exp_name} ({label1} vs. {label2})")
    results = run_simulation(exp["agent1"], exp["agent2"],
                             exp["params1"], exp["params2"],
                             StochasticRPSGame, episodes=episodes, trials=trials)
    
    # (a) Moving Average Rewards
    avg_rewards1 = np.mean(results['rewards_agent1'], axis=0)
    avg_rewards2 = np.mean(results['rewards_agent2'], axis=0)
    ma1 = moving_average(avg_rewards1, window=100)
    ma2 = moving_average(avg_rewards2, window=100)
    plt.figure(figsize=(12,6))
    plt.plot(ma1, label=label1)
    plt.plot(ma2, label=label2)
    plt.title(f"Moving Average Rewards ({label1} vs. {label2})")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{exp_name}_rewards.png"))
    plt.close()
    # Save processed moving average data
    for i, val in enumerate(ma1):
        export_moving_avg.append({
            'experiment': exp_name,
            'agent_name': label1,
            'episode': i + 100,  # offset by window size
            'moving_avg_reward': val
        })
    for i, val in enumerate(ma2):
        export_moving_avg.append({
            'experiment': exp_name,
            'agent_name': label2,
            'episode': i + 100,
            'moving_avg_reward': val
        })
    
    # (b) Cumulative Scores
    avg_cum1 = np.mean(results['cum_scores_agent1'], axis=0)
    avg_cum2 = np.mean(results['cum_scores_agent2'], axis=0)
    plt.figure(figsize=(12,6))
    plt.plot(avg_cum1, label=label1)
    plt.plot(avg_cum2, label=label2)
    plt.title(f"Cumulative Scores ({label1} vs. {label2})")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Score")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{exp_name}_cumulative.png"))
    plt.close()
    for i, val in enumerate(avg_cum1):
        export_cumulative.append({
            'experiment': exp_name,
            'agent_name': label1,
            'episode': i,
            'cumulative_score': val
        })
    for i, val in enumerate(avg_cum2):
        export_cumulative.append({
            'experiment': exp_name,
            'agent_name': label2,
            'episode': i,
            'cumulative_score': val
        })
    
    # (c) Environment State Evolution
    avg_states = np.mean(results['states'], axis=0)
    plt.figure(figsize=(12,6))
    plt.plot(avg_states)
    plt.title(f"Average Environment State ({label1} vs. {label2})")
    plt.xlabel("Episode")
    plt.ylabel("State")
    plt.savefig(os.path.join(output_dir, f"{exp_name}_states.png"))
    plt.close()
    for i, state_val in enumerate(avg_states):
        export_states.append({
            'experiment': exp_name,
            'episode': i,
            'average_state': state_val
        })
    
    # (d) Joint Action Frequency Heatmap
    joint_actions_all = np.concatenate(results['joint_actions'])
    heatmap_data = np.zeros((n_actions, n_actions))
    for (a1, a2) in joint_actions_all:
        heatmap_data[a1, a2] += 1
    heatmap_data /= np.sum(heatmap_data)
    plt.figure(figsize=(8,6))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu",
                xticklabels=['Rock', 'Paper', 'Scissors'],
                yticklabels=['Rock', 'Paper', 'Scissors'])
    plt.title(f"Joint Action Frequency ({label1} vs. {label2})")
    plt.xlabel(f"{label2} Action")
    plt.ylabel(f"{label1} Action")
    plt.savefig(os.path.join(output_dir, f"{exp_name}_joint_actions.png"))
    plt.close()
    for i in range(heatmap_data.shape[0]):
        for j in range(heatmap_data.shape[1]):
            export_joint_actions.append({
                'experiment': exp_name,
                'action_agent1': i,
                'action_agent2': j,
                'frequency': heatmap_data[i, j]
            })
    
    # (e) Reward Distribution (using error bars data)
    mean_reward1 = np.mean(results['rewards_agent1'], axis=0)
    std_reward1 = np.std(results['rewards_agent1'], axis=0)
    x_axis = np.arange(len(mean_reward1))
    plt.figure(figsize=(12,6))
    plt.errorbar(x_axis[::50], mean_reward1[::50], yerr=std_reward1[::50],
                 fmt='o', capsize=3, label=label1)
    plt.title(f"Reward Distribution ({label1} vs. {label2})")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{exp_name}_reward_distribution.png"))
    plt.close()
    for i in range(0, len(mean_reward1), 50):
        export_reward_distribution.append({
            'experiment': exp_name,
            'agent_name': label1,
            'episode': x_axis[i],
            'mean_reward': mean_reward1[i],
            'std_reward': std_reward1[i]
        })
    mean_reward2 = np.mean(results['rewards_agent2'], axis=0)
    std_reward2 = np.std(results['rewards_agent2'], axis=0)
    for i in range(0, len(mean_reward2), 50):
        export_reward_distribution.append({
            'experiment': exp_name,
            'agent_name': label2,
            'episode': x_axis[i],
            'mean_reward': mean_reward2[i],
            'std_reward': std_reward2[i]
        })
    
    # (f) Policy Evolution (if available)
    extra1 = results['extra_info_agent1']
    if extra1 and extra1[0]['policy_history'] is not None:
        policies = [np.array(trial) for trial in [ex['policy_history'] for ex in extra1]]
        avg_policy = np.mean(np.array(policies), axis=0)
        ep_axis = np.arange(avg_policy.shape[0])
        plt.figure(figsize=(12,6))
        plt.stackplot(ep_axis, avg_policy.T, labels=['Rock', 'Paper', 'Scissors'])
        plt.title(f"{label1} Policy Evolution")
        plt.xlabel("Episode")
        plt.ylabel("Action Probability")
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(output_dir, f"{exp_name}_{label1.replace(' ', '_').lower()}_policy_evolution.png"))
        plt.close()
        for i, row in enumerate(avg_policy):
            export_policy_evolution.append({
                'experiment': exp_name,
                'agent_name': label1,
                'episode': i,
                'rock': row[0],
                'paper': row[1],
                'scissors': row[2]
            })
    extra2 = results['extra_info_agent2']
    if extra2 and extra2[0]['policy_history'] is not None:
        policies = [np.array(trial) for trial in [ex['policy_history'] for ex in extra2]]
        avg_policy = np.mean(np.array(policies), axis=0)
        ep_axis = np.arange(avg_policy.shape[0])
        plt.figure(figsize=(12,6))
        plt.stackplot(ep_axis, avg_policy.T, labels=['Rock', 'Paper', 'Scissors'])
        plt.title(f"{label2} Policy Evolution")
        plt.xlabel("Episode")
        plt.ylabel("Action Probability")
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(output_dir, f"{exp_name}_{label2.replace(' ', '_').lower()}_policy_evolution.png"))
        plt.close()
        for i, row in enumerate(avg_policy):
            export_policy_evolution.append({
                'experiment': exp_name,
                'agent_name': label2,
                'episode': i,
                'rock': row[0],
                'paper': row[1],
                'scissors': row[2]
            })
    
    # (g) Epsilon Decay (if available)
    if extra1 and extra1[0]['epsilon_history'] is not None:
        epsilons = [np.array(trial) for trial in [ex['epsilon_history'] for ex in extra1]]
        avg_epsilon = np.mean(np.array(epsilons), axis=0)
        plt.figure(figsize=(12,6))
        plt.plot(avg_epsilon, label=f"{label1} Epsilon")
        plt.title(f"Epsilon Decay ({label1})")
        plt.xlabel("Episode")
        plt.ylabel("Epsilon")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"{exp_name}_{label1.replace(' ', '_').lower()}_epsilon_decay.png"))
        plt.close()
        for i, eps in enumerate(avg_epsilon):
            export_epsilon_decay.append({
                'experiment': exp_name,
                'agent_name': label1,
                'episode': i,
                'epsilon': eps
            })
    if extra2 and extra2[0]['epsilon_history'] is not None:
        epsilons = [np.array(trial) for trial in [ex['epsilon_history'] for ex in extra2]]
        avg_epsilon = np.mean(np.array(epsilons), axis=0)
        plt.figure(figsize=(12,6))
        plt.plot(avg_epsilon, label=f"{label2} Epsilon")
        plt.title(f"Epsilon Decay ({label2})")
        plt.xlabel("Episode")
        plt.ylabel("Epsilon")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"{exp_name}_{label2.replace(' ', '_').lower()}_epsilon_decay.png"))
        plt.close()
        for i, eps in enumerate(avg_epsilon):
            export_epsilon_decay.append({
                'experiment': exp_name,
                'agent_name': label2,
                'episode': i,
                'epsilon': eps
            })
    
    # (h) Q-Value Evolution and Convergence (if available)
    if 'q_history_agent1' in results and len(results['q_history_agent1']) > 0:
        q_history_list = results['q_history_agent1']
        max_values_list = []
        norm_diff_list = []
        for trial in q_history_list:
            trial_max = []
            trial_norm_diff = []
            prev = None
            for snapshot in trial:
                if snapshot.ndim == 1:
                    value = np.max(snapshot)
                else:
                    value, _ = solve_minimax(snapshot)
                trial_max.append(value)
                if prev is not None:
                    trial_norm_diff.append(np.linalg.norm(snapshot - prev))
                else:
                    trial_norm_diff.append(0)
                prev = snapshot
            max_values_list.append(np.array(trial_max))
            norm_diff_list.append(np.array(trial_norm_diff))
        avg_max_values = pad_and_average(max_values_list)
        avg_norm_diff = pad_and_average(norm_diff_list)
        plt.figure(figsize=(12,6))
        plt.plot(avg_max_values, label=f"{label1} Q-Value")
        plt.title(f"Q-Value Evolution ({label1})")
        plt.xlabel("Update Index (visits to state 0)")
        plt.ylabel("Value")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"{exp_name}_{label1.replace(' ', '_').lower()}_qvalues.png"))
        plt.close()
        plt.figure(figsize=(12,6))
        plt.plot(avg_norm_diff)
        plt.title(f"Q-Value Convergence ({label1})")
        plt.xlabel("Update Index (visits to state 0)")
        plt.ylabel("Norm Difference")
        plt.savefig(os.path.join(output_dir, f"{exp_name}_{label1.replace(' ', '_').lower()}_qvalue_convergence.png"))
        plt.close()
        for i, val in enumerate(avg_max_values):
            export_q_evolution.append({
                'experiment': exp_name,
                'agent_name': label1,
                'update_index': i,
                'q_value': val
            })
        for i, val in enumerate(avg_norm_diff):
            export_q_convergence.append({
                'experiment': exp_name,
                'agent_name': label1,
                'update_index': i,
                'norm_diff': val
            })
    if 'q_history_agent2' in results and len(results['q_history_agent2']) > 0:
        q_history_list = results['q_history_agent2']
        max_values_list = []
        norm_diff_list = []
        for trial in q_history_list:
            trial_max = []
            trial_norm_diff = []
            prev = None
            for snapshot in trial:
                if snapshot.ndim == 1:
                    value = np.max(snapshot)
                else:
                    value, _ = solve_minimax(snapshot)
                trial_max.append(value)
                if prev is not None:
                    trial_norm_diff.append(np.linalg.norm(snapshot - prev))
                else:
                    trial_norm_diff.append(0)
                prev = snapshot
            max_values_list.append(np.array(trial_max))
            norm_diff_list.append(np.array(trial_norm_diff))
        avg_max_values = pad_and_average(max_values_list)
        avg_norm_diff = pad_and_average(norm_diff_list)
        plt.figure(figsize=(12,6))
        plt.plot(avg_max_values, label=f"{label2} Q-Value")
        plt.title(f"Q-Value Evolution ({label2})")
        plt.xlabel("Update Index (visits to state 0)")
        plt.ylabel("Value")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"{exp_name}_{label2.replace(' ', '_').lower()}_qvalues.png"))
        plt.close()
        plt.figure(figsize=(12,6))
        plt.plot(avg_norm_diff)
        plt.title(f"Q-Value Convergence ({label2})")
        plt.xlabel("Update Index (visits to state 0)")
        plt.ylabel("Norm Difference")
        plt.savefig(os.path.join(output_dir, f"{exp_name}_{label2.replace(' ', '_').lower()}_qvalue_convergence.png"))
        plt.close()
        for i, val in enumerate(avg_max_values):
            export_q_evolution.append({
                'experiment': exp_name,
                'agent_name': label2,
                'update_index': i,
                'q_value': val
            })
        for i, val in enumerate(avg_norm_diff):
            export_q_convergence.append({
                'experiment': exp_name,
                'agent_name': label2,
                'update_index': i,
                'norm_diff': val
            })
    
    # --- Compute wins for this experiment ---
    # For each trial, compare final cumulative scores.
    for trial in range(trials):
        final_score1 = results['cum_scores_agent1'][trial][-1]
        final_score2 = results['cum_scores_agent2'][trial][-1]
        if final_score1 > final_score2:
            total_wins[label1] += 1
        elif final_score2 > final_score1:
            total_wins[label2] += 1
        # Ties are not counted.

    print(f"Experiment {exp_name} completed. Plots saved in '{output_dir}' directory.\n")

############################################
# Plot Total Wins Bar Chart (Tournament Winner)
############################################
plt.figure(figsize=(8,6))
agents = list(total_wins.keys())
wins = list(total_wins.values())
plt.bar(agents, wins, color=['blue', 'green', 'red', 'purple'])
plt.title("Total Wins per Agent in Tournament")
plt.xlabel("Agent")
plt.ylabel("Wins")
plt.savefig(os.path.join(output_dir, "total_wins.png"))
plt.close()
print("Total wins bar plot saved as 'total_wins.png'.")

############################################
# Export Processed Data to CSV Files
############################################
pd.DataFrame(export_moving_avg).to_csv("rps_processed_moving_avg.csv", index=False)
pd.DataFrame(export_cumulative).to_csv("rps_processed_cumulative.csv", index=False)
pd.DataFrame(export_states).to_csv("rps_processed_states.csv", index=False)
pd.DataFrame(export_joint_actions).to_csv("rps_processed_joint_actions.csv", index=False)
pd.DataFrame(export_reward_distribution).to_csv("rps_processed_reward_distribution.csv", index=False)
pd.DataFrame(export_policy_evolution).to_csv("rps_processed_policy_evolution.csv", index=False)
pd.DataFrame(export_epsilon_decay).to_csv("rps_processed_epsilon_decay.csv", index=False)
pd.DataFrame(export_q_evolution).to_csv("rps_processed_q_evolution.csv", index=False)
pd.DataFrame(export_q_convergence).to_csv("rps_processed_q_convergence.csv", index=False)

print("Exported processed data to CSV files.")
print("All experiments are complete.")
