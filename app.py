import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
flights = pd.read_csv("https://raw.githubusercontent.com/byuidatascience/data4python4ds/master/data-raw/flights/flights.csv")
airlines = pd.read_csv("https://raw.githubusercontent.com/byuidatascience/data4python4ds/master/data-raw/airlines/airlines.csv")
planes = pd.read_csv("https://raw.githubusercontent.com/byuidatascience/data4python4ds/master/data-raw/planes/planes.csv")
weather = pd.read_csv("https://raw.githubusercontent.com/byuidatascience/data4python4ds/master/data-raw/weather/weather.csv")

# =============================================================================
# DATA PREPARATION (Maintaining Notebook Logic)
# =============================================================================

# Load data
df_flights = flights
df_planes = planes
df_weather = weather
df_airlines = airlines

# 1. Logic: Identify Cancelled Flights (Selection Bias Analysis)
df_flights['is_cancelled'] = df_flights['dep_time'].isna()

# 2. Logic: Fleet Join (To get manufacturer and year of plane)
df_fleet = pd.merge(df_flights, df_planes, on='tailnum', how='left', suffixes=('', '_plane'))

# 3. Logic: Weather Join 
# Joining on year, month, day, hour, and origin airport
df_weather_merged = pd.merge(df_flights, df_weather, on=['year', 'month', 'day', 'hour', 'origin'])

# 4. Logic: SFO Performance
df_sfo = df_flights[df_flights['dest'] == 'SFO']

# 5. Logic: Finance Challenge (Yield Curve Inversion Logic)
# Using 'ME' to avoid the FutureWarning
dates = pd.date_range(start='2010-01-01', end='2023-12-31', freq='ME')
yield_data = pd.DataFrame({
    'date': dates,
    'spread': np.sin(np.linspace(0, 10, len(dates))) + np.random.normal(0, 0.1, len(dates))
})
yield_data['recession'] = yield_data['spread'] < 0

# =============================================================================
# APP SETUP & STYLING
# =============================================================================

app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.CYBORG],
    suppress_callback_exceptions=True
)

SIDEBAR_STYLE = {
    "position": "fixed", "top": 0, "left": 0, "bottom": 0,
    "width": "20rem", "padding": "2rem 1rem", "background-color": "#111", "color": "white"
}

CONTENT_STYLE = {
    "margin-left": "22rem", "margin-right": "2rem", "padding": "2rem 1rem"
}

sidebar = html.Div(
    [
        html.H2("NYC Flights '13", className="display-6", style={'color': '#00d4ff'}),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("1. Selection Bias", href="/", active="exact"),
                dbc.NavLink("2. Fleet & Age", href="/fleet", active="exact"),
                dbc.NavLink("3. SFO & Confounding", href="/sfo", active="exact"),
                dbc.NavLink("4. Weather Impact", href="/weather", active="exact"),
                dbc.NavLink("5. Simpson's Paradox", href="/paradox", active="exact"),
                dbc.NavLink("6. Delay Dynamics", href="/delays", active="exact"),
                dbc.NavLink("7. Finance Challenge", href="/finance", active="exact"),
            ],
            vertical=True, pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", style=CONTENT_STYLE)
app.layout = html.Div([dcc.Location(id="url"), sidebar, content])

# =============================================================================
# PAGE LAYOUTS (Visualizations)
# =============================================================================

# --- PAGE 1: OPERATIONS & CANCELLATIONS ---
def layout_selection_bias():
    return html.Div([
        html.H3("Cancelled Flights & Selection Bias"),
        html.P("Analyzing if missing data (cancellations) skews delay averages."),
        dcc.Graph(id='cancelled-graph')
    ])

# --- PAGE 2: FLEET ---
def layout_fleet():
    return html.Div([
        html.H3("Manufacturer Performance & Plane Age"),
        dbc.Row([
            dbc.Col(dcc.Graph(id='fleet-treemap'), width=6),
            dbc.Col(dcc.Graph(id='age-delay-scatter'), width=6)
        ])
    ])

# --- PAGE 3: SFO ---
def layout_sfo():
    return html.Div([
        html.H3("SFO Route: Carrier Performance vs Route Difficulty"),
        dcc.Graph(id='sfo-graph')
    ])

# --- PAGE 4: WEATHER ---
def layout_weather():
    return html.Div([
        html.H3("Weather vs. Departure Delay"),
        dcc.Dropdown(
            id='weather-metric',
            options=[{'label': k, 'value': k} for k in ['visib', 'precip', 'wind_speed', 'temp']],
            value='visib', className='mb-3', style={'color': 'black'}
        ),
        dcc.Graph(id='weather-graph')
    ])

# --- PAGE 5: SIMPSON'S PARADOX ---
def layout_paradox():
    return html.Div([
        html.H3("Simpson's Paradox: Time of Day vs Carrier"),
        html.P("Watch how carrier performance rankings flip based on the hour of departure."),
        dcc.Graph(id='paradox-graph')
    ])

# --- PAGE 6: ARRIVAL VS DEPARTURE ---
def layout_delays():
    return html.Div([
        html.H3("Can Pilots 'Make Up' Time?"),
        html.P("Arrival vs Departure Delay colored by Flight Duration (Air Time)"),
        dcc.Graph(id='duration-graph')
    ])

# --- PAGE 7: FINANCE ---
def layout_finance():
    return html.Div([
        html.H3("Finance: Yield Curve Inversion"),
        html.P("Predicting Recessions via the 10Y-2Y Spread"),
        dcc.Graph(id='yield-graph')
    ])

# =============================================================================
# CALLBACKS
# =============================================================================

@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def display_page(pathname):
    if pathname == "/": return layout_selection_bias()
    if pathname == "/fleet": return layout_fleet()
    if pathname == "/sfo": return layout_sfo()
    if pathname == "/weather": return layout_weather()
    if pathname == "/paradox": return layout_paradox()
    if pathname == "/delays": return layout_delays()
    if pathname == "/finance": return layout_finance()
    return html.H1("404 Not Found")

# 1. Selection Bias Callback
@app.callback(Output('cancelled-graph', 'figure'), Input('url', 'pathname'))
def update_cancelled(_):
    data = df_flights.groupby(['month', 'origin'])['is_cancelled'].mean().reset_index()
    fig = px.line(data, x='month', y='is_cancelled', color='origin', markers=True, 
                  title="Cancellation Rate by Origin (Source of Selection Bias)")
    fig.update_layout(template="plotly_dark", yaxis_title="Probability of Cancellation")
    return fig

# 2. Fleet Callback
@app.callback([Output('fleet-treemap', 'figure'), Output('age-delay-scatter', 'figure')], Input('url', 'pathname'))
def update_fleet(_):
    # Treemap
    m_counts = df_fleet.groupby('manufacturer').size().reset_index(name='count')
    tree = px.treemap(m_counts[m_counts['count']>10], path=['manufacturer'], values='count', title="Manufacturers by Flight Vol")
    
    # Age vs Delay
    age_data = df_fleet.groupby('year_plane')['arr_delay'].mean().reset_index()
    scatter = px.scatter(age_data, x='year_plane', y='arr_delay', trendline="ols", title="Plane Age vs. Mean Delay")
    
    tree.update_layout(template="plotly_dark")
    scatter.update_layout(template="plotly_dark")
    return tree, scatter

# 3. SFO Callback
@app.callback(Output('sfo-graph', 'figure'), Input('url', 'pathname'))
def update_sfo(_):
    sfo_perf = df_sfo.groupby('carrier')['arr_delay'].mean().sort_values().reset_index()
    fig = px.bar(sfo_perf, x='carrier', y='arr_delay', color='arr_delay', title="Average Delay on SFO Route by Carrier")
    fig.update_layout(template="plotly_dark")
    return fig

# 4. Weather Callback
@app.callback(Output('weather-graph', 'figure'), Input('weather-metric', 'value'))
def update_weather(metric):
    # Grouping to avoid over-plotting (300k points)
    w_data = df_weather_merged.groupby(metric)['dep_delay'].mean().reset_index()
    fig = px.scatter(w_data, x=metric, y='dep_delay', trendline="lowess", title=f"Impact of {metric} on Delays")
    fig.update_layout(template="plotly_dark")
    return fig

# 5. Paradox Callback
@app.callback(Output('paradox-graph', 'figure'), Input('url', 'pathname'))
def update_paradox(_):
    data = df_flights.groupby(['carrier', 'hour'])['arr_delay'].mean().reset_index()
    fig = px.line(data, x='hour', y='arr_delay', color='carrier', title="Delay by Hour of Day (Simpson's Paradox Drilldown)")
    fig.update_layout(template="plotly_dark")
    return fig

# 6. Delays Callback
@app.callback(Output('duration-graph', 'figure'), Input('url', 'pathname'))
def update_delays(_):
    # Sampling for performance
    sample = df_flights.dropna(subset=['arr_delay', 'dep_delay', 'air_time']).sample(5000)
    fig = px.scatter(sample, x='dep_delay', y='arr_delay', color='air_time',
                     opacity=0.5, title="Dep Delay vs Arr Delay (Color = Air Time)")
    fig.add_shape(type="line", x0=0, y0=0, x1=200, y1=200, line=dict(color="Red", dash="dash"))
    fig.update_layout(template="plotly_dark")
    return fig

# 7. Yield Curve Callback
@app.callback(Output('yield-graph', 'figure'), Input('url', 'pathname'))
def update_yield(_):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=yield_data['date'], y=yield_data['spread'], name='Spread', line=dict(color='#00d4ff')))
    fig.add_hline(y=0, line_dash="dash", line_color="white")
    
    # Highlight Inversion zones (Recessions)
    recessions = yield_data[yield_data['recession']]
    for d in recessions['date']:
        fig.add_vrect(x0=d, x1=d, fillcolor="red", opacity=0.1, layer="below", line_width=0)
        
    fig.update_layout(template="plotly_dark", title="10Y-2Y Yield Spread (Inversions in Red)", yaxis_title="Spread (%)")
    return fig

# =============================================================================
# RUN APP
# =============================================================================

server = app.server
if __name__ == "__main__":
    app.run(debug=False, port=8050)
