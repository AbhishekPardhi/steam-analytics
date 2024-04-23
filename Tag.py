import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# Load dataset
df = pd.read_csv("data/games.csv")  # Replace "your_dataset.csv" with your dataset file path


df['Genres'] = df['Genres'].fillna('')

# Apply lambda function
df['Genres'] = df['Genres'].apply(lambda x: x.split(',') if isinstance(x, str) else [])

# Clean Tags column
df['Tags'] = df['Tags'].str.replace(';', ',')
df['Tags'] = df['Tags'].str.split(',')

# Add sales and platform count 
def extract_max(d):
    return int(d.split('-')[1])

df['Total revenue'] = df['Price'] * df['Estimated owners'].apply(extract_max)
df['Platform count'] = df['Windows'].astype(int) + df['Mac'].astype(int) + df['Linux'].astype(int)


# Create Dash app
app = dash.Dash(__name__)

# Define app layout
app.layout = html.Div([
    dcc.Dropdown(
        id='genre-dropdown',
        options=[{'label': genre, 'value': genre} for genre in df['Genres'].explode().unique()],
        value=df['Genres'].explode().unique()[0],
        multi=False
    ),
    dcc.Graph(id='tag-bar-chart'),
    dcc.Graph(id='engagement-correlation'),
    dcc.Graph(id='sales-correlation')
])

# Define callback to update tag bar chart based on selected genre
@app.callback(
    Output('tag-bar-chart', 'figure'),
    [Input('genre-dropdown', 'value')]
)
def update_tag_bar_chart(selected_genre):
    # Filter dataframe based on selected genre
    genre_df = df[df['Genres'].apply(lambda x: selected_genre in x)]

    # Flatten list of tags associated with games in the selected genre
    tags_list = [tag for sublist in genre_df['Tags'].tolist() if isinstance(sublist, list) for tag in sublist]

    # Calculate tag frequencies
    tag_counts = pd.Series(tags_list).value_counts()

    # Create bar chart
    fig = px.bar(tag_counts.head(10), x=tag_counts.head(10).index, y=tag_counts.head(10).values,
                 labels={'x': 'Tag', 'y': 'Frequency'}, title=f"Top 10 Tags for Genre: {selected_genre}")

    return fig

# Explore the correlation between player engagement metrics (such as playtime, and number of sessions) and game genres to understand which genres attract more dedicated players.

@app.callback(
    Output('engagement-correlation', 'figure'),
    [Input('genre-dropdown', 'value')]
)
def update_engagement_correlation(selected_genre):
    ddf = df.explode('Genres')
    genre_engagement = ddf.groupby('Genres')['Average playtime forever'].mean().reset_index()
    
    # Create scatter plot
    fig = px.scatter(genre_engagement, x='Genres', y='Average playtime forever', title=f"Engagement Correlation for Genre: {selected_genre}")
   
    # mark selected genre differently
    fig.add_trace(px.scatter(genre_engagement[genre_engagement['Genres'] == selected_genre], x='Genres', y='Average playtime forever').data[0])
    fig.update_traces(marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')),
                  selector=dict(mode='markers'))
    return fig


@app.callback(
    Output('sales-correlation', 'figure'),
    [Input('genre-dropdown', 'value')]

)
def update_sales_correlation(selected_genre):
    # do it for a single genre
    ddf = df.explode('Genres')
    ddf = ddf[ddf['Genres'] == selected_genre]

    # sum of total revenue for each platform count
    platform_revenue = ddf.groupby('Platform count')['Total revenue'].sum().reset_index()

    # average playtime for each platform count
    time_spent = ddf.groupby('Platform count')['Average playtime forever'].mean().reset_index()

    # subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Sales Correlation", "Engagement Correlation"), specs=[[{"type": "pie"}, {"type": "pie"}]])

    # Sales Correlation
    fig.add_trace(px.pie(platform_revenue, values='Total revenue', names='Platform count').data[0], row=1, col=1)

    # Engagement Correlation
    fig.add_trace(px.pie(time_spent, values='Average playtime forever', names='Platform count').data[0], row=1, col=2)
    fig.update_layout(title_text=f"Correlation based on availability across Windows, Mac and Linux for genre: {selected_genre}")

    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
