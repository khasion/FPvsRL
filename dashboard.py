import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import numpy as np
import os
import ast

# Function to load processed CSV data for a given game (rps or mp)
def load_processed_data(game):
    data = {}
    metrics = [
        "moving_avg", "cumulative", "states", "joint_actions",
        "reward_distribution", "policy_evolution", "epsilon_decay",
        "q_evolution", "q_convergence"
    ]
    for metric in metrics:
        filename = f"{game.lower()}_processed_{metric}.csv"
        if os.path.exists(filename):
            data[metric] = pd.read_csv(filename)
        else:
            data[metric] = pd.DataFrame()
    return data

# Load processed data for both games.
rps_data = load_processed_data("rps")
mp_data = load_processed_data("mp")
# Dictionary mapping game selection to processed data.
processed_data = {"RPS": rps_data, "MP": mp_data}

# (Optional) Also load the raw simulation data (if available for additional plots)
def load_simulation_data():
    files = ['rps_simulation_data.csv', 'mp_simulation_data.csv']
    dataframes = []
    for file in files:
        if os.path.exists(file):
            df = pd.read_csv(file)
            for col in ['policy_distribution', 'q_values']:
                df[col] = df[col].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else None)
            if 'rps' in file.lower():
                df['game'] = 'RPS'
            elif 'mp' in file.lower():
                df['game'] = 'MP'
            dataframes.append(df)
        else:
            print(f"Warning: {file} not found.")
    if dataframes:
        simulation_data = pd.concat(dataframes, ignore_index=True)
    else:
        simulation_data = pd.DataFrame()
    return simulation_data

simulation_data = load_simulation_data()

# For the scenario dropdown, use processed cumulative data (if available)
if not processed_data["RPS"]["cumulative"].empty and "experiment" in processed_data["RPS"]["cumulative"].columns:
    scenario_options = [{'label': exp, 'value': exp} 
                        for exp in processed_data["RPS"]["cumulative"]["experiment"].unique()]
    default_scenario = scenario_options[0]['value']
else:
    scenario_options = []
    default_scenario = None

# Dictionary to map experiment names to agent labels.
experiment_labels = {
    "fp_vs_ql": ("Fictitious Play", "Q–Learning"),
    "fp_vs_bp": ("Fictitious Play", "Belief–Based"),
    "ql_vs_mm": ("Q–Learning", "Minimax RL"),
    "fp_vs_mm": ("Fictitious Play", "Minimax RL"),
    "ql_vs_bp": ("Q–Learning", "Belief–Based"),
    "mm_vs_bp": ("Minimax RL", "Belief–Based"),
}

# Initialize Dash app.
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CERULEAN])

# Layout: Controls and six graphs.
app.layout = html.Div(
    style={'backgroundColor': '#1e1e2e', 'color': 'white', 'padding': '20px'},
    children=[
        # Controls
        html.Div([
            html.Div([
                html.Label('Scenario', style={'fontSize': '24px'}),
                dcc.Dropdown(
                    id='scenario-dropdown',
                    options=scenario_options,
                    value=default_scenario,
                    clearable=False,
                    style={'width': '90%', 'color': 'black'}
                )
            ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            
            html.Div([
                html.Label('Game Selection', style={'fontSize': '24px'}),
                dcc.RadioItems(
                    id='game-selection',
                    options=[
                        {'label': ' Rock-Paper-Scissors (RPS)', 'value': 'RPS'},
                        {'label': ' Matching Pennies (MP)', 'value': 'MP'}
                    ],
                    value='RPS',
                    labelStyle={'display': 'block', 'fontSize': '20px'}
                )
            ], style={'width': '20%', 'display': 'inline-block'}),
            
            html.Div([
                html.Label('Episodes', style={'fontSize': '24px'}),
                dcc.Slider(
                    id='episodes-slider',
                    min=500,
                    max=5000,
                    step=500,
                    value=5000,
                    marks={i: str(i) for i in range(500, 5000, 500)},
                )
            ], style={'width': '40%', 'display': 'inline-block'}),
        ]),
        html.Hr(style={'borderColor': 'gray'}),
        
        # Graphs arranged in two columns per row.
        html.Div([ dcc.Graph(id='cumulative-value-chart') ], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([ dcc.Graph(id='moving-average-chart') ], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([ dcc.Graph(id='convergence-chart') ], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([ dcc.Graph(id='confusion-matrix-heatmap') ], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([ dcc.Graph(id='reward-distribution-chart') ], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([ dcc.Graph(id='policy-evolution-chart') ], style={'width': '48%', 'display': 'inline-block'}),
    ]
)

# Callback 1: Cumulative Score Chart (using processed cumulative CSV)
@app.callback(
    Output('cumulative-value-chart', 'figure'),
    [Input('scenario-dropdown', 'value'),
     Input('game-selection', 'value'),
     Input('episodes-slider', 'value')]
)
def update_cumulative_chart(selected_scenario, selected_game, max_episode):
    df = processed_data[selected_game]["cumulative"]
    if df.empty:
        return px.line(title="No Cumulative Data Available")
    df = df[(df['experiment'] == selected_scenario) & (df['episode'] <= max_episode)]
    if df.empty:
        return px.line(title="No Data for Selected Experiment")
    fig = px.line(df, x='episode', y='cumulative_score', color='agent_name',
                  labels={'cumulative_score': 'Cumulative Score', 'episode': 'Episode', 'agent_name': 'Agent'},
                  title='Cumulative Score over Episodes')
    fig.update_layout(template='plotly_dark')
    return fig

# Callback 2: Moving Average Rewards Chart (new replacement graph)
@app.callback(
    Output('moving-average-chart', 'figure'),
    [Input('scenario-dropdown', 'value'),
     Input('game-selection', 'value'),
     Input('episodes-slider', 'value')]
)
def update_moving_average_chart(selected_scenario, selected_game, max_episode):
    df = processed_data[selected_game]["moving_avg"]
    if df.empty:
        return px.line(title="No Moving Average Data Available")
    df = df[(df['experiment'] == selected_scenario) & (df['episode'] <= max_episode)]
    if df.empty:
        return px.line(title="No Data for Selected Experiment")
    fig = px.line(df, x='episode', y='moving_avg_reward', color='agent_name',
                  labels={'moving_avg_reward': 'Moving Average Reward', 'episode': 'Episode', 'agent_name': 'Agent'},
                  title='Moving Average Rewards over Episodes')
    fig.update_layout(template='plotly_dark')
    return fig

# Callback 3: Convergence Chart (using processed q_convergence CSV)
@app.callback(
    Output('convergence-chart', 'figure'),
    [Input('scenario-dropdown', 'value'),
     Input('game-selection', 'value'),
     Input('episodes-slider', 'value')]
)
def update_convergence_chart(selected_scenario, selected_game, max_episode):
    df = processed_data[selected_game]["q_convergence"]
    if df.empty:
        return px.line(title="No Q–Convergence Data Available")
    df = df[(df['experiment'] == selected_scenario) & (df['update_index'] <= max_episode)]
    if df.empty:
        return px.line(title="No Data for Selected Experiment")
    fig = px.line(df, x='update_index', y='norm_diff', color='agent_name',
                  labels={'norm_diff': 'Average Norm Difference', 'update_index': 'Update Index'},
                  title='Q–Value Convergence Over Episodes')
    fig.update_layout(template='plotly_dark')
    return fig

# Callback 4: Joint Action Frequency Heatmap (with agent names on the axes)
@app.callback(
    Output('confusion-matrix-heatmap', 'figure'),
    [Input('scenario-dropdown', 'value'),
     Input('game-selection', 'value'),
     Input('episodes-slider', 'value')]
)
def update_heatmap(selected_scenario, selected_game, max_episode):
    df = processed_data[selected_game]["joint_actions"]
    if df.empty:
        return px.imshow(np.zeros((2,2)), text_auto=True, title="No Joint Action Data Available")
    df = df[df['experiment'] == selected_scenario]
    if df.empty:
        return px.imshow(np.zeros((2,2)), text_auto=True, title="No Data for Selected Experiment")
    
    # Use experiment_labels mapping to get agent names.
    if selected_scenario in experiment_labels:
        agent1_label, agent2_label = experiment_labels[selected_scenario]
    else:
        agent1_label, agent2_label = "Agent1", "Agent2"
    
    pivot = df.pivot(index='action_agent1', columns='action_agent2', values='frequency').fillna(0)
    fig = px.imshow(pivot, text_auto=True, 
                    labels=dict(x=f"{agent2_label} Action", y=f"{agent1_label} Action", color="Frequency"),
                    title="Joint Action Frequency Heatmap")
    fig.update_layout(template='plotly_dark')
    return fig

# Callback 5: Reward Distribution Chart (using processed reward_distribution CSV)
@app.callback(
    Output('reward-distribution-chart', 'figure'),
    [Input('scenario-dropdown', 'value'),
     Input('game-selection', 'value'),
     Input('episodes-slider', 'value')]
)
def update_reward_distribution(selected_scenario, selected_game, max_episode):
    df = processed_data[selected_game]["reward_distribution"]
    if df.empty:
        return px.box(title="No Reward Distribution Data Available")
    df = df[(df['experiment'] == selected_scenario) & (df['episode'] <= max_episode)]
    if df.empty:
        return px.box(title="No Data for Selected Experiment")
    fig = px.box(df, x='agent_name', y='mean_reward', points="all",
                 labels={'mean_reward': 'Mean Reward', 'agent_name': 'Agent'},
                 title="Reward Distribution")
    fig.update_layout(template='plotly_dark')
    return fig

# Callback 6: Policy Evolution Chart (new replacement graph)
@app.callback(
    Output('policy-evolution-chart', 'figure'),
    [Input('scenario-dropdown', 'value'),
     Input('game-selection', 'value'),
     Input('episodes-slider', 'value')]
)
def update_policy_evolution_chart(selected_scenario, selected_game, max_episode):
    df = processed_data[selected_game]["policy_evolution"]
    if df.empty:
        return px.line(title="No Policy Evolution Data Available")
    df = df[(df['experiment'] == selected_scenario) & (df['episode'] <= max_episode)]
    if df.empty:
        return px.line(title="No Data for Selected Experiment")
    
    # Depending on the game, determine which columns to plot.
    if selected_game == "RPS":
        value_vars = ['rock', 'paper', 'scissors']
    else:  # Matching Pennies
        value_vars = ['heads', 'tails']
        
    # Melt the dataframe to long format.
    df_melted = df.melt(id_vars=['experiment', 'agent_name', 'episode'], 
                        value_vars=value_vars,
                        var_name='action', value_name='probability')
    
    # Create an area chart with facets by agent.
    fig = px.area(df_melted, x='episode', y='probability', color='action', facet_col='agent_name',
                  labels={'probability': 'Action Probability', 'episode': 'Episode', 'action': 'Action'},
                  title='Policy Evolution over Episodes')
    fig.update_layout(template='plotly_dark')
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
