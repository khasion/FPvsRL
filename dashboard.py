import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import numpy as np
import json
import ast
import os

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

# (Optional) Also load the raw simulation data (if needed for additional plots)
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

# Initialize Dash app.
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CERULEAN])

# Layout: Dropdowns, radio items, slider, and six graphs.
app.layout = html.Div(
    style={'backgroundColor': '#1e1e2e', 'color': 'white', 'padding': '20px'},
    children=[
        # Controls
        html.Div([
            html.Div([
                html.Label('Scenario', style={'fontSize': '24px'}),
                dcc.Dropdown(
                    id='scenario-dropdown',
                    options=[{'label': exp, 'value': exp} for exp in simulation_data['experiment'].unique()],
                    value=simulation_data['experiment'].unique()[0] if not simulation_data.empty else None,
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
        html.Div([ dcc.Graph(id='win-percentage-pie') ], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([ dcc.Graph(id='convergence-chart') ], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([ dcc.Graph(id='confusion-matrix-heatmap') ], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([ dcc.Graph(id='reward-distribution-chart') ], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([ dcc.Graph(id='epsilon-decay-chart') ], style={'width': '48%', 'display': 'inline-block'}),
    ]
)

# Callback 1: Cumulative Score Chart (from processed cumulative CSV)
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

# Callback 2: Win Percentage Pie Chart (using processed cumulative CSV with trial info)
@app.callback(
    Output('win-percentage-pie', 'figure'),
    [Input('scenario-dropdown', 'value'),
     Input('game-selection', 'value'),
     Input('episodes-slider', 'value')]
)
def update_win_percentage(selected_scenario, selected_game, max_episode):
    df = processed_data[selected_game]["cumulative"]
    if df.empty or 'trial' not in df.columns:
        return px.pie(title="No Trial-level Cumulative Data Available")
    # Filter for the last episode.
    df = df[(df['experiment'] == selected_scenario) & (df['episode'] == max_episode)]
    if df.empty:
        return px.pie(title="No Data for Selected Experiment")
    pivot = df.pivot(index='trial', columns='agent_name', values='cumulative_score')
    if pivot.shape[1] < 2:
        return px.pie(title="Not enough agent data")
    winners = pivot.apply(lambda row: row.idxmax(), axis=1)
    win_counts = winners.value_counts().reset_index()
    win_counts.columns = ['agent_name', 'wins']
    fig = px.pie(win_counts, values='wins', names='agent_name', title='Win Percentage by Agent')
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

# Callback 4: Joint Action Frequency Heatmap (using processed joint_actions CSV)
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
    pivot = df.pivot(index='action_agent1', columns='action_agent2', values='frequency').fillna(0)
    fig = px.imshow(pivot, text_auto=True, 
                    labels=dict(x="Agent 2 Action", y="Agent 1 Action", color="Frequency"),
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

# Callback 6: Epsilon Decay Chart (using processed epsilon_decay CSV)
@app.callback(
    Output('epsilon-decay-chart', 'figure'),
    [Input('scenario-dropdown', 'value'),
     Input('game-selection', 'value'),
     Input('episodes-slider', 'value')]
)
def update_epsilon_decay(selected_scenario, selected_game, max_episode):
    df = processed_data[selected_game]["epsilon_decay"]
    if df.empty:
        return px.line(title="No Epsilon Data Available")
    df = df[(df['experiment'] == selected_scenario) & (df['episode'] <= max_episode)]
    if df.empty:
        return px.line(title="No Data for Selected Experiment")
    fig = px.line(df, x='episode', y='epsilon', color='agent_name',
                  labels={'epsilon': 'Epsilon', 'episode': 'Episode'},
                  title='Epsilon Decay Over Episodes')
    fig.update_layout(template='plotly_dark')
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
