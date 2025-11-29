"""
This module contains functions for data visualization using Plotly and Matplotlib.
It creates interactive and static charts for exploring the MovieLens dataset,
covering distributions, time-series, correlations, hierarchies, and text analysis.
"""
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud


def plot_rating_distribution(ratings_df, user_id=None):
    """
    Plots an interactive histogram of rating distributions.
    Can be filtered for a specific user.
    """
    data = ratings_df
    title = "Global Rating Distribution"
    if user_id:
        data = ratings_df[ratings_df['userId'] == user_id]
        title = f"Rating Distribution for User {user_id}"

    if data.empty:
        return None  # Return None if user has no ratings to plot

    fig = px.histogram(data, x="rating", nbins=10,
                       title=f"<b>{title}</b>",
                       labels={'rating': 'Rating Stars'},
                       color_discrete_sequence=['#FF4B4B'])
    fig.update_layout(bargap=0.1)
    return fig


def plot_activity_over_time(ratings_df, user_id=None):
    """
    Plots an interactive area chart of user rating activity over time.
    Can be filtered for a specific user.
    """
    df = ratings_df.copy()
    title = "User Activity Over Time (Monthly)"
    if user_id:
        df = df[df['userId'] == user_id]
        title = f"Activity Over Time for User {user_id}"

    if df.empty:
        return None

    df['date'] = pd.to_datetime(df['timestamp'], unit='s')
    daily_counts = df.resample('M', on='date').size().reset_index(name='count')

    fig = px.area(daily_counts, x='date', y='count',
                  title=f"<b>{title}</b>",
                  markers=True)
    return fig


def plot_genre_treemap(movies_df):
    """
    Plots an interactive treemap of movie genres.
    """
    movies_df['main_genre'] = movies_df['genres'].apply(
        lambda x: x.split('|')[0] if pd.notnull(x) else 'Unknown')
    counts = movies_df['main_genre'].value_counts().reset_index()
    counts.columns = ['genre', 'count']

    fig = px.treemap(counts, path=['genre'], values='count',
                     title="<b>Movie Genres Hierarchy (Treemap)</b>",
                     color='count', color_continuous_scale='viridis')
    return fig


def plot_wordcloud(tags_df):
    """
    Generates a static word cloud from user-generated tags.
    """
    if tags_df is None or tags_df.empty:
        return None
    text = " ".join(t for t in tags_df['tag'].dropna())
    if not text:
        return None

    wc = WordCloud(width=800, height=400, background_color='black',
                   colormap='Pastel1').generate(text)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    return fig


def plot_popularity_vs_quality(ratings_df):
    """
    Plots an interactive scatter plot analyzing movie popularity vs. quality.
    """
    stats = ratings_df.groupby('movieId').agg(
        {'rating': ['mean', 'count']}).reset_index()
    stats.columns = ['movieId', 'avg_rating', 'vote_count']
    stats = stats[stats['vote_count'] > 30]  # Filter noise

    fig = px.scatter(stats, x='vote_count', y='avg_rating',
                     trendline="ols",  # Ordinary Least Squares regression line
                     title="<b>Popularity vs. Quality Analysis</b>",
                     opacity=0.5,
                     labels={'vote_count': 'Number of Ratings',
                             'avg_rating': 'Average Score'})
    return fig


def plot_genre_violin(movies_df, ratings_df, user_id=None):
    """
    Plots an interactive violin chart showing rating distribution per genre.
    Can be filtered for a specific user.
    """
    data = pd.merge(movies_df, ratings_df, on='movieId')
    title = "<b>Rating Distribution by Genre</b>"
    if user_id:
        data = data[data['userId'] == user_id]
        title = f"<b>Genre Rating Distribution for User {user_id}</b>"

    if data.empty:
        return None

    # Explode genres to have one genre per row
    data['genre'] = data['genres'].str.split('|')
    data = data.explode('genre')

    fig = px.violin(data, x='genre', y='rating',
                    title=title,
                    labels={'genre': 'Genre', 'rating': 'Rating'},
                    color='genre')
    return fig
