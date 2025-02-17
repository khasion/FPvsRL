# Zero-Sum Games Learning Algorithms

This repository contains Python scripts for simulating learning agents in two-person zero-sum games. The experiments are run in two games:
- **Stochastic Rock-Paper-Scissors (RPS)**
- **Matching Pennies (MP)**

The implemented agents include:
- Fictitious Play (FP)
- Q-Learning (QL)
- Minimax RL
- Belief-Based

Additionally, an interactive dashboard built with Dash and Plotly is provided to visualize and explore the experimental results.

## Prerequisites

Ensure you have Python 3.7+ installed. Install the required packages by running:

```bash
pip install numpy matplotlib scipy seaborn tqdm dash dash-bootstrap-components plotly pandas
```

# Running the Simulation Scripts

```bash
python rock_paper_scissors.py
python matching_pennies.py
```

# Runing the Dashboard

```bash
python dashboard.py
```