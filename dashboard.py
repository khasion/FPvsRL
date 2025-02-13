import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
import json
import os

# Function to load simulation data from CSV files.
def load_simulation_data():
    files = ['rps_simulation_data.csv', 'mp_simulation_data.csv']
    dataframes = []
    
    for file in files:
        if os.path.exists(file):
            df = pd.read_csv(file)
            # Tag data by game type.
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

# Load the data once.
simulation_data = load_simulation_data()

# Initialize Dash app.
app = dash.Dash(__name__)

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
                    style={'width': '90%'}
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
                    min=10,
                    max=100,
                    step=10,
                    value=50,
                    marks={i: str(i) for i in range(10, 110, 20)},
                )
            ], style={'width': '40%', 'display': 'inline-block'}),
        ]),
        html.Hr(style={'borderColor': 'gray'}),
        
        # Graphs arranged in two columns per row.
        html.Div([
            dcc.Graph(id='cumulative-value-chart')
        ], style={'width': '48%', 'display': 'inline-block'}),
        
        html.Div([
            dcc.Graph(id='win-percentage-pie')
        ], style={'width': '48%', 'display': 'inline-block'}),
        
        html.Div([
            dcc.Graph(id='convergence-chart')
        ], style={'width': '48%', 'display': 'inline-block'}),
        
        html.Div([
            dcc.Graph(id='confusion-matrix-heatmap')
        ], style={'width': '48%', 'display': 'inline-block'}),
        
        html.Div([
            dcc.Graph(id='reward-distribution-chart')
        ], style={'width': '48%', 'display': 'inline-block'}),
        
        html.Div([
            dcc.Graph(id='epsilon-decay-chart')
        ], style={'width': '48%', 'display': 'inline-block'}),
    ]
)

# ------------------- Callback 1: Cumulative Score Chart -------------------
@app.callback(
    Output('cumulative-value-chart', 'figure'),
    [Input('scenario-dropdown', 'value'),
     Input('game-selection', 'value'),
     Input('episodes-slider', 'value')]
)
def update_cumulative_chart(selected_scenario, selected_game, max_episode):
    df = simulation_data[
        (simulation_data['experiment'] == selected_scenario) &
        (simulation_data['game'] == selected_game) &
        (simulation_data['episode'] <= max_episode)
    ]
    if df.empty:
        return px.line(title="No Data Available")
    # Sort by trial and episode; compute cumulative reward per trial and agent.
    df = df.sort_values(['trial', 'episode'])
    df['cumulative_score'] = df.groupby(['trial', 'agent_name'])['reward'].cumsum()
    # Average cumulative score across trials by agent and episode.
    avg_df = df.groupby(['agent_name', 'episode'], as_index=False)['cumulative_score'].mean()
    fig = px.line(avg_df, x='episode', y='cumulative_score', color='agent_name',
                  labels={'cumulative_score': 'Cumulative Score', 'episode': 'Episode', 'agent_name': 'Agent'},
                  title='Cumulative Score over Episodes')
    fig.update_layout(template='plotly_dark')
    return fig

# ------------------- Callback 2: Win Percentage Pie Chart -------------------
@app.callback(
    Output('win-percentage-pie', 'figure'),
    [Input('scenario-dropdown', 'value'),
     Input('game-selection', 'value'),
     Input('episodes-slider', 'value')]
)
def update_win_percentage(selected_scenario, selected_game, max_episode):
    df = simulation_data[
        (simulation_data['experiment'] == selected_scenario) &
        (simulation_data['game'] == selected_game) &
        (simulation_data['episode'] <= max_episode)
    ]
    if df.empty:
        return px.pie(title="No Data Available")
    df = df.sort_values(['trial', 'episode'])
    df['cumulative_score'] = df.groupby(['trial', 'agent_name'])['reward'].cumsum()
    # Get final cumulative score for each trial and agent.
    final_scores = df.groupby(['trial', 'agent_name'], as_index=False).last()
    # Pivot so that each trial has both agents' final scores.
    pivot = final_scores.pivot(index='trial', columns='agent_name', values='cumulative_score')
    if pivot.shape[1] < 2:
        return px.pie(title="Not enough agent data")
    # Determine winner per trial.
    winners = pivot.apply(lambda row: row.idxmax(), axis=1)
    win_counts = winners.value_counts().reset_index()
    win_counts.columns = ['agent_name', 'wins']
    fig = px.pie(win_counts, values='wins', names='agent_name', title='Win Percentage by Agent')
    fig.update_layout(template='plotly_dark')
    return fig

# ------------------- Callback 3: Convergence Chart -------------------
@app.callback(
    Output('convergence-chart', 'figure'),
    [Input('scenario-dropdown', 'value'),
     Input('game-selection', 'value'),
     Input('episodes-slider', 'value')]
)
def update_convergence_chart(selected_scenario, selected_game, max_episode):
    # We use Q–values (if available) to compute the norm difference between consecutive snapshots.
    df = simulation_data[
        (simulation_data['experiment'] == selected_scenario) &
        (simulation_data['game'] == selected_game) &
        (simulation_data['episode'] <= max_episode)
    ]
    if df.empty or 'q_values' not in df.columns:
        return px.line(title="No Q–value Data Available")
    
    convergence_data = []
    grouped = df.groupby(['trial', 'agent_name'])
    for (trial, agent), group in grouped:
        group = group.sort_values('episode')
        prev = None
        for _, row in group.iterrows():
            q_json = row['q_values']
            if pd.isna(q_json):
                continue
            try:
                q_val = np.array(json.loads(q_json))
            except Exception:
                continue
            if prev is not None:
                diff = np.linalg.norm(q_val - prev)
                convergence_data.append({'trial': trial, 'agent_name': agent, 'episode': row['episode'], 'norm_diff': diff})
            prev = q_val
    if not convergence_data:
        return px.line(title="No Convergence Data Available")
    conv_df = pd.DataFrame(convergence_data)
    avg_conv = conv_df.groupby(['agent_name', 'episode'], as_index=False)['norm_diff'].mean()
    fig = px.line(avg_conv, x='episode', y='norm_diff', color='agent_name',
                  labels={'norm_diff': 'Average Norm Difference', 'episode': 'Episode'},
                  title='Q–Value Convergence Over Episodes')
    fig.update_layout(template='plotly_dark')
    return fig

# ------------------- Callback 4: Joint Action Frequency Heatmap -------------------
@app.callback(
    Output('confusion-matrix-heatmap', 'figure'),
    [Input('scenario-dropdown', 'value'),
     Input('game-selection', 'value'),
     Input('episodes-slider', 'value')]
)
def update_heatmap(selected_scenario, selected_game, max_episode):
    df = simulation_data[
        (simulation_data['experiment'] == selected_scenario) &
        (simulation_data['game'] == selected_game) &
        (simulation_data['episode'] <= max_episode)
    ]
    if df.empty:
        return px.imshow(np.zeros((2,2)), text_auto=True, title="No Data Available")
    # Pivot the data: each trial and episode should have two rows (one per agent).
    pivot = df.pivot_table(index=['experiment', 'trial', 'episode'], columns='agent_name', values='action', aggfunc='first').reset_index()
    if pivot.shape[1] < 3:
        return px.imshow(np.zeros((2,2)), text_auto=True, title="Not enough agent data")
    # Assume the first two agent columns.
    agent_cols = pivot.columns[3:5]
    joint_actions = pivot[agent_cols]
    actions = np.sort(df['action'].unique())
    freq_matrix = np.zeros((len(actions), len(actions)))
    for _, row in joint_actions.iterrows():
        a1 = int(row[agent_cols[0]])
        a2 = int(row[agent_cols[1]])
        freq_matrix[a1, a2] += 1
    if freq_matrix.sum() > 0:
        freq_matrix = freq_matrix / freq_matrix.sum()
    fig = px.imshow(freq_matrix, 
                    labels=dict(x=agent_cols[1], y=agent_cols[0], color="Frequency"),
                    x=[str(a) for a in actions],
                    y=[str(a) for a in actions],
                    title="Joint Action Frequency Heatmap")
    fig.update_layout(template='plotly_dark')
    return fig

# ------------------- Callback 5: Reward Distribution Chart -------------------
@app.callback(
    Output('reward-distribution-chart', 'figure'),
    [Input('scenario-dropdown', 'value'),
     Input('game-selection', 'value'),
     Input('episodes-slider', 'value')]
)
def update_reward_distribution(selected_scenario, selected_game, max_episode):
    df = simulation_data[
        (simulation_data['experiment'] == selected_scenario) &
        (simulation_data['game'] == selected_game) &
        (simulation_data['episode'] <= max_episode)
    ]
    if df.empty:
        return px.box(title="No Data Available")
    fig = px.box(df, x='agent_name', y='reward', points="all",
                 labels={'reward': 'Reward', 'agent_name': 'Agent'},
                 title="Reward Distribution")
    fig.update_layout(template='plotly_dark')
    return fig

# ------------------- Callback 6: Epsilon Decay Chart -------------------
@app.callback(
    Output('epsilon-decay-chart', 'figure'),
    [Input('scenario-dropdown', 'value'),
     Input('game-selection', 'value'),
     Input('episodes-slider', 'value')]
)
def update_epsilon_decay(selected_scenario, selected_game, max_episode):
    df = simulation_data[
        (simulation_data['experiment'] == selected_scenario) &
        (simulation_data['game'] == selected_game) &
        (simulation_data['episode'] <= max_episode)
    ]
    if df.empty or 'epsilon' not in df.columns:
        return px.line(title="No Epsilon Data Available")
    avg_eps = df.groupby(['agent_name', 'episode'], as_index=False)['epsilon'].mean()
    fig = px.line(avg_eps, x='episode', y='epsilon', color='agent_name',
                  labels={'epsilon': 'Epsilon', 'episode': 'Episode'},
                  title='Epsilon Decay Over Episodes')
    fig.update_layout(template='plotly_dark')
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
