import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
import numpy as np
import glob
import os

# Function to load simulation data from CSV files.
def load_simulation_data():
    files = ['rps_simulation_data.csv', 'mp_simulation_data.csv']
    dataframes = []
    
    for file in files:
        if os.path.exists(file):
            df = pd.read_csv(file)
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

# Layout with Dropdowns and Graphs.
app.layout = html.Div(style={'backgroundColor': '#1e1e2e', 'color': 'white', 'padding': '20px'}, children=[
    html.Div([
        html.Label('Scenario', style={'fontSize': '24px'}),
        dcc.Dropdown(
            id='scenario-dropdown',
            options=[{'label': exp, 'value': exp} for exp in simulation_data['experiment'].unique()],
            value=simulation_data['experiment'].unique()[0],
            clearable=False,
            style={'width': '80%'}
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
    ], style={'width': '48%', 'display': 'inline-block'})
])

# Example callback for cumulative chart.
@app.callback(
    dash.Output('cumulative-value-chart', 'figure'),
    [dash.Input('scenario-dropdown', 'value'),
     dash.Input('game-selection', 'value')]
)
def update_cumulative_chart(selected_scenario, selected_game):
    filtered_data = simulation_data[
        (simulation_data['experiment'] == selected_scenario) &
        (simulation_data['game'] == selected_game)
    ]
    agent1, agent2 = filtered_data[['agent1', 'agent2']].iloc[0]
    fig = px.line(filtered_data, x='episode', y=['cumulative_score_agent1', 'cumulative_score_agent2'],
                  labels={'value': 'Cumulative Score', 'episode': 'Episode'},
                  title=f'Cumulative Value: {agent1} vs {agent2} ({selected_game})')
    fig.update_layout(template='plotly_dark')
    return fig

# Define additional callbacks similarly...
# (win-percentage-pie, convergence-chart, confusion-matrix-heatmap)

if __name__ == '__main__':
    app.run_server(debug=True)
