import pandas as pd
import plotly.express as px
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
    dcc.Graph(id='tag-bar-chart')
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

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
