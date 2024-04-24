import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import random

def read_data(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['Genres'])
    df_original = df.copy()

    # Preprocess the data: Calculate average value for 'Estimated owners' range
    df['Estimated owners'] = df['Estimated owners'].str.replace('[^\d-]+', '', regex=True)
    df[['Min Owners', 'Max Owners']] = df['Estimated owners'].str.split('-', expand=True).astype(float)
    df['Average Owners'] = (df['Min Owners'] + df['Max Owners']) / 2

    # Convert Positive & Negative Reviews to %
    df['% Positive Reviews'] = 100 * df['Positive']/(df['Positive'] + df['Negative'])
    df['% Negative Reviews'] = 100 - df['% Positive Reviews']

    # Release date preprocess
    df['Release date'] = pd.to_datetime(df['Release date'], errors='coerce')
    # Handling the case where some dates are in "Month Year" format
    # Assuming the release date to be the first day of the month
    df['Release date'] = df['Release date'].fillna(pd.to_datetime(df['Release date'], errors='coerce', format='%b %Y'))
    df['Month_Year'] = df['Release date'].dt.to_period('M').astype(str)

    # Duplicate rows for each genre
    genre_list = df['Genres'].str.split(',', expand=True).stack().unique()
    df['Genres'] = df['Genres'].str.split(',')
    df = df.explode('Genres')
    
    return df, df_original, genre_list

df, df_original, genre_list = read_data('data/games.csv')
df_tech = pd.read_csv('data/tech_adv.csv')
df_tech['Year'] = pd.to_datetime(df_tech['Year'], format='%Y')
merged_df = pd.merge(df, df_tech, left_on='Release date', right_on='Year', how='left')

app = dash.Dash(__name__)

@app.callback(
    Output('demand-histogram', 'figure'),
    [Input('genre-dropdown-1', 'value')]
)
def price_sensitivity_visualization_1(genre):
    genre_data = df[df['Genres'] == genre]
    fig = px.histogram(genre_data, x='Price', y='Average Owners', nbins=30, 
                       title=f'Average Owners Distribution for {genre}', 
                       labels={'Price': 'Price Range', 'Average Owners': 'Average Owners'},
                       histfunc='avg')
    return fig

def timeline_visualization(df):
    # Group by month and year and count the number of games released each month
    monthly_counts = df.groupby('Month_Year').size().reset_index(name='Count')

    # Calculate rolling average for smoothing
    rolling_avg_window = 12
    monthly_counts['Smoothed_Count'] = monthly_counts['Count'].rolling(window=rolling_avg_window, min_periods=1).mean()

    fig = px.line(monthly_counts, x='Month_Year', y=['Count', 'Smoothed_Count'], 
                labels={'Count': 'Number of Games Released', 'Month_Year': 'Release Month'},
                title='Number of Steam Games Released Each Month')

    fig.update_xaxes(title_text='Release Month', range=[monthly_counts['Month_Year'].min(), monthly_counts['Month_Year'].max()])

    fig.update_yaxes(title_text='Number of Games Released')
    
    return fig

@app.callback(
    Output('game-timeline', 'figure'),
    [Input('genre-dropdown-3', 'value')]
)

def tech_adv_timeline(selected_genre):
    # Define color map for technological advancements
    adv_colors = {}

    # Generate random color for each technological advancement
    for advancement in df_tech['Advancement'].unique():
        adv_colors[advancement] = "#{:06x}".format(random.randint(0, 0xFFFFFF))

    filtered_df = merged_df[merged_df['Genres'] == selected_genre]

    # Group by month and year and count the number of games released each month
    monthly_counts = filtered_df.groupby('Month_Year').size().reset_index(name='Count')

    fig = px.line(monthly_counts, x='Month_Year', y='Count', 
                  labels={'Count': 'Number of Games Released', 'Month_Year': 'Release Month'},
                  title=f'Number of {selected_genre} Games Released Each Month')
    
    # Add colored segments for technological advancements
    for index, row in df_tech.iterrows():
        year = row['Year']
        advancement = row['Advancement']
        color = adv_colors.get(advancement, 'rgb(0, 0, 0)')  # Default color is black if not found in the map
        fig.add_trace(go.Scatter(x=[year, year], y=[0, monthly_counts['Count'].max()], mode='lines', line=dict(color=color), name=advancement))

    # Set x-axis range
    fig.update_xaxes(title_text='Release Month', range=[monthly_counts['Month_Year'].min(), monthly_counts['Month_Year'].max()])
    
    fig.update_yaxes(title_text='Number of Games Released')
    
    return fig

def peak_ccu_visualisation(df):
    # Group by genres and calculate the average Peak CCU for each genre
    genre_peak_ccu = df.groupby('Genres')['Peak CCU'].mean().reset_index()

    # Plot the correlation heatmap
    fig = px.bar(genre_peak_ccu, x='Genres', y='Peak CCU', title='Average Peak CCU by Genre')
    
    return fig

def genre_correlation_visualization(df):
    df = pd.read_csv('data/games.csv')
    
    # Remove rows with NaN values in the 'Genres' column
    df = df.dropna(subset=['Genres'])

    # Split genres into lists
    df['Genres'] = df['Genres'].str.split(',')

    # Create a dictionary to store the count of each genre
    genre_count = {}

    # Iterate through each row in the dataframe to count the occurrence of each genre
    for index, row in df.iterrows():
        for genre in row['Genres']:
            if genre in genre_count:
                genre_count[genre] += 1
            else:
                genre_count[genre] = 1

    # Extract the list of genres
    genres = list(genre_count.keys())

    # Create an empty dataframe to store the sum of Peak CCU for each genre combination
    genre_combinations_sum = pd.DataFrame(0, index=genres, columns=genres, dtype=float)

    # Iterate through each row in the dataframe to calculate the sum of Peak CCU for each genre combination
    for index, row in df.iterrows():
        for genre_1 in row['Genres']:
            for genre_2 in row['Genres']:
                if genre_1 != genre_2:
                    genre_combinations_sum.loc[genre_1, genre_2] += row['Peak CCU']

    # Create an empty dataframe to store the average Peak CCU for each genre combination
    genre_combinations_avg = genre_combinations_sum.copy()

    # Calculate the average Peak CCU for each genre combination
    for i, genre_1 in enumerate(genres):
        for j, genre_2 in enumerate(genres):
            if genre_1 != genre_2:
                count = min(genre_count[genre_1], genre_count[genre_2])  # Minimum count of games contributing to both genres
                genre_combinations_avg.iloc[i, j] /= count

    # Replace NaN values with 0
    genre_combinations_avg = genre_combinations_avg.fillna(0)

    # Plot the heatmap
    fig = px.imshow(genre_combinations_avg, x=genres, y=genres, color_continuous_scale='viridis', title='Average Peak CCU by Genre Combination')
    fig.update_layout(xaxis_title='Genre 1', yaxis_title='Genre 2')
    
    return fig

@app.callback(
    Output('tag-bar-chart', 'figure'),
    [Input('genre-dropdown-2', 'value')]
)
def tag_visualization(selected_genre):
    df = df_original.copy()
    df['Genres'] = df['Genres'].fillna('')

    # Apply lambda function
    df['Genres'] = df['Genres'].apply(lambda x: x.split(',') if isinstance(x, str) else [])

    # Clean Tags column
    df['Tags'] = df['Tags'].str.replace(';', ',')
    df['Tags'] = df['Tags'].str.split(',')
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

def price_sensitivity_visualization_2(df):
    cf = df[(df['Average playtime forever'] < 25000) & (df['Price'] < 100) & (df['Peak CCU'] < 32000)]

    fig = px.scatter(cf, x='Average playtime forever', y='Peak CCU', color='Price', color_continuous_scale='inferno',
                    size_max=40, hover_name=cf.index, title='Scatter Plot for all Games')

    fig.update_layout(coloraxis_colorbar=dict(title='Price'))
    
    return fig

@app.callback(
    Output('wordcloud-graph', 'figure'),
    [Input('genre-dropdown-4', 'value')]
)
def wordcloud_visualization(genre):
    # df = pd.read_csv('data/games.csv')
    # df = df.dropna(subset=['Genres'])
    # Filter the dataframe for the selected genre
    genre_df = df_original[df_original['Genres'].str.contains(genre, na=False)]
    
    # Concatenate the descriptions
    text = ' '.join(genre_df['About the game'].dropna())
    
    # Create a word cloud figure using Plotly Express
    wordcloud = WordCloud(background_color="black", colormap="viridis", prefer_horizontal=0.9, max_font_size=40).generate_from_text(text)
    wordcloud_fig = px.imshow(wordcloud, title=f'Word Cloud for {genre}')
    wordcloud_fig.update_xaxes(showline=False, visible=False, ticks='')
    wordcloud_fig.update_yaxes(showline=False, visible=False, ticks='')
    wordcloud_fig.update_layout(coloraxis_showscale=False)
    
    return wordcloud_fig

@app.callback(
    Output('review-playtime-plot', 'figure'),
    [Input('genre-dropdown-5', 'value'),
     Input('review-type', 'value')]
)

def game_review_visualization(selected_genre, review_type):
    # Threshold to filter relevant games
    df_copy = df[(df['Positive'] + df['Negative'] >= 5000) & (df['Average playtime forever']>=500)]

    filtered_df = df_copy[df_copy['Genres'] == selected_genre]
    
    # Determine x-axis and title based on review type selection
    y_column = review_type
    # Create scatter plot
    fig = px.scatter(filtered_df, x='Average playtime forever', y=y_column,
                     hover_name='Name', size_max=60,
                     labels={y_column: f"{review_type}", "Average playtime forever": "Average Playtime (Forever)"})
    
    fig.update_layout(title=f"{review_type} vs Average Playtime for {selected_genre} Games",
                      yaxis_title=f"{review_type}",
                      xaxis_title="Average Playtime (Forever)")

    return fig

timeline_fig = timeline_visualization(df)
peak_ccu_fig = peak_ccu_visualisation(df)
genre_correlation_fig = genre_correlation_visualization(df_original)
# genre_correlation_fig = peak_ccu_visualisation(df)
price_sensitivity_fig = price_sensitivity_visualization_2(df)

app.layout = html.Div([
    html.Div([
        html.Div([
            html.H2('Price Sensitivity Visualization 1'),
            dcc.Dropdown(
                id='genre-dropdown-1',
                options=[{'label': genre, 'value': genre} for genre in genre_list],
                value=genre_list[0],
                clearable=False
            ),
            dcc.Graph(id='demand-histogram')
        ], className='six columns'),
        html.Div([
            html.H2('Price Sensitivity Visualization 2'),
            dcc.Graph(
                id='graph5',
                figure=price_sensitivity_fig
            ),
        ], className='six columns')
    ], className='row'),
    html.Div([
        html.Div([
            html.H2('Peak CCU Visualization'),
            dcc.Graph(
                id='graph3',
                figure=peak_ccu_fig
            ),
        ], className='six columns'),
        html.Div([
            html.H2('Genre Correlation Visualization'),
            dcc.Graph(
                id='graph4',
                figure=genre_correlation_fig
            ),
        ], className='six columns')
    ], className='row'),
    html.Div([
        html.Div([
            html.H2('Tags Visualization'),
            dcc.Dropdown(
                id='genre-dropdown-2',
                options=[{'label': genre, 'value': genre} for genre in genre_list],
                value=genre_list[0],
                multi=False
            ),
            dcc.Graph(id='tag-bar-chart')
        ], className='six columns'),
        html.Div([
            html.H2('Timeline Visualization'),
            dcc.Graph(
                id='graph2',
                figure=timeline_fig
            ),
        ], className='six columns')
    ], className='row'),
    html.Div([
        html.Div([
            html.H2('Tech Advancements and Release Timeline Visualization by Genre'),
            dcc.Dropdown(
                id='genre-dropdown-3',
                options=[{'label': genre, 'value': genre} for genre in genre_list],
                value=genre_list[0], 
                multi=False,
            ),
            dcc.Graph(id='game-timeline')
        ], className='twelve columns')
    ], className='row'),
    html.Div([
        html.Div([
            html.H2("Word Cloud Generator for Game Descriptions by Genre"),
            dcc.Dropdown(
                id='genre-dropdown-4',
                options=[{'label': genre, 'value': genre} for genre in df_original['Genres'].unique()],
                value=df_original['Genres'].unique()[0],
                clearable=False,
                style={'width': '50%'}
            ),
            dcc.Graph(id='wordcloud-graph')
        ], className='six columns'),
        html.Div([
            html.H2("Game Review vs Playtime Analysis by Genre"),
            dcc.Dropdown(
                id='genre-dropdown-5',
                options=[{'label': genre, 'value': genre} for genre in genre_list],
                value=genre_list[0],  
                multi=False,
                style={'width': '50%'}
            ),
            dcc.RadioItems(
                id='review-type',
                options=[
                    {'label': '% Negative Reviews', 'value': '% Negative Reviews'},
                    {'label': '% Positive Reviews', 'value': '% Positive Reviews'}
                ],
                value='% Negative Reviews',
                inline=True,
                style={'width': '50%'}
            ),
            dcc.Graph(id='review-playtime-plot')
        ], className='six columns')
    ], className='row'),
])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
