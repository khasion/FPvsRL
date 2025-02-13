import dash
from dash import dcc, html
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Read the exported CSV data
simulation_data = pd.read_csv("rps_simulation_data.csv")

# Dash App Initialization
app = dash.Dash(__name__)

# Layout
app.layout = html.Div(
    style={'backgroundColor': '#1e1e2e', 'color': 'white', 'padding': '20px'},
    children=[
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
    ]
)

# Callbacks for updating the charts
@app.callback(
    dash.Output('cumulative-value-chart', 'figure'),
    dash.Input('scenario-dropdown', 'value'),
    dash.Input('game-selection', 'value')
)
def update_cumulative_chart(selected_scenario, selected_game):
    filtered_data = simulation_data[
        (simulation_data['experiment'] == selected_scenario) &
        (simulation_data['game'] == selected_game)
    ]
    # Get agent names from the first row
    agent1, agent2 = filtered_data[['agent1', 'agent2']].iloc[0]
    fig = px.line(
        filtered_data,
        x='episode',
        y=['cumulative_score_agent1', 'cumulative_score_agent2'],
        labels={'value': 'Cumulative Score', 'episode': 'Episode'},
        title=f'Cumulative Value: {agent1} vs {agent2} ({selected_game})'
    )
    fig.update_layout(template='plotly_dark')
    return fig

@app.callback(
    dash.Output('win-percentage-pie', 'figure'),
    dash.Input('scenario-dropdown', 'value'),
    dash.Input('game-selection', 'value')
)
def update_pie_chart(selected_scenario, selected_game):
    filtered_data = simulation_data[
        (simulation_data['experiment'] == selected_scenario) &
        (simulation_data['game'] == selected_game)
    ]
    agent1, agent2 = filtered_data[['agent1', 'agent2']].iloc[0]
    # Count wins for each agent (assuming positive reward indicates a win)
    win_counts = filtered_data[['reward_agent1', 'reward_agent2']].apply(lambda x: (x > 0).sum(), axis=0)
    pie_data = pd.DataFrame({
        'labels': [f'Wins {agent1}', f'Wins {agent2}'],
        'values': win_counts.values
    })
    fig = px.pie(
        pie_data, names='labels', values='values',
        title=f'Win Percentage ({selected_game})'
    )
    fig.update_layout(template='plotly_dark')
    return fig

@app.callback(
    dash.Output('convergence-chart', 'figure'),
    dash.Input('scenario-dropdown', 'value'),
    dash.Input('game-selection', 'value')
)
def update_convergence_chart(selected_scenario, selected_game):
    filtered_data = simulation_data[
        (simulation_data['experiment'] == selected_scenario) &
        (simulation_data['game'] == selected_game)
    ]
    agent1, agent2 = filtered_data[['agent1', 'agent2']].iloc[0]
    fig = px.line(
        filtered_data,
        x='episode',
        y=['reward_agent1', 'reward_agent2'],
        labels={'value': 'Reward', 'episode': 'Episode'},
        title=f'Convergence: {agent1} vs {agent2} ({selected_game})'
    )
    fig.update_layout(template='plotly_dark')
    return fig

@app.callback(
    dash.Output('confusion-matrix-heatmap', 'figure'),
    dash.Input('scenario-dropdown', 'value'),
    dash.Input('game-selection', 'value')
)
def update_confusion_matrix(selected_scenario, selected_game):
    filtered_data = simulation_data[
        (simulation_data['experiment'] == selected_scenario) &
        (simulation_data['game'] == selected_game)
    ]
    agent1, agent2 = filtered_data[['agent1', 'agent2']].iloc[0]
    # Create a simple confusion matrix; this is just a demonstration and might need adjustment
    confusion_matrix = np.array([
        [filtered_data['reward_agent1'].mean(), filtered_data['reward_agent2'].mean()],
        [filtered_data['reward_agent2'].mean(), filtered_data['reward_agent1'].mean()]
    ])
    fig = px.imshow(
        confusion_matrix,
        color_continuous_scale='Blues',
        title=f'Confusion Matrix: {agent1} vs {agent2} ({selected_game})'
    )
    fig.update_layout(template='plotly_dark')
    return fig

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
