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
    # Work on a copy so we don't mutate the caller's DataFrame
    mcopy = movies_df.copy()
    mcopy['main_genre'] = mcopy['genres'].apply(
        lambda x: x.split('|')[0] if pd.notnull(x) else 'Unknown')
    counts = mcopy['main_genre'].value_counts().reset_index()
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


def plot_wordcloud_by_year(tags_df, movies_df, year):
    """Generate a wordcloud for tags associated with movies released in a given year.

    tags_df: DataFrame with at least ['movieId', 'tag']
    movies_df: DataFrame with at least ['movieId', 'year']
    year: integer year to filter by
    """
    if tags_df is None or tags_df.empty or movies_df is None or movies_df.empty:
        return None

    try:
        # Ensure numeric year in movies
        m = movies_df.copy()
        if 'year' not in m.columns:
            return None
        m = m.dropna(subset=['year'])
        m['year'] = m['year'].astype(int)
    except Exception:
        return None

    # Get movieIds for the year
    mids = m[m['year'] == int(year)]['movieId'].unique()
    if len(mids) == 0:
        return None

    # Aggregate tags for those movie ids
    t = tags_df[tags_df['movieId'].isin(mids)]
    if t is None or t.empty:
        return None

    text = " ".join(t['tag'].dropna().astype(str).tolist())
    if not text:
        return None

    wc = WordCloud(width=800, height=400, background_color='black',
                   colormap='Pastel1').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    fig.suptitle(f"Popular tags — {year}", color='white')
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


def plot_metrics_summary(metrics: dict, title: str = "Model Metrics Summary"):
    """Create a bar chart summarizing the model metrics in a friendly way.

    metrics expects a mapping of display_name -> numeric_value.
    Only numeric values are kept. Non-numeric values are ignored.
    """
    if metrics is None or len(metrics) == 0:
        return None

    # Filter numeric keys
    numeric_items = {k: v for k,
                     v in metrics.items() if isinstance(v, (int, float))}
    if not numeric_items:
        return None

    # Sort for stable presentation
    items = sorted(numeric_items.items(), key=lambda x: x[0])
    labels, vals = zip(*items)

    # Build a Plotly bar chart
    fig = px.bar(x=list(labels), y=list(vals), title=f"<b>{title}</b>",
                 labels={'x': 'Metric', 'y': 'Value'}, color=list(vals),
                 color_continuous_scale='viridis')
    fig.update_layout(showlegend=False, height=420)
    return fig


def plot_metrics_radar(metrics: dict, title: str = "Model Radar Overview"):
    """Create a radar (polar) chart for a set of metrics.

    The function expects metrics to be a mapping of label->value where
    values are numerics in the range [0,1] or convertible to that range.
    Non-numeric values will be ignored.
    """
    if metrics is None or len(metrics) == 0:
        return None

    # Keep only numeric entries
    numeric_items = {k: float(v) for k, v in metrics.items()
                     if isinstance(v, (int, float))}
    if not numeric_items:
        return None

    labels = list(numeric_items.keys())
    values = [numeric_items[k] for k in labels]

    # Normalize values to [0,1] if any are larger than 1 (e.g., RMSE)
    max_val = max(values) if values else 1.0
    if max_val > 1.0:
        values = [v / max_val for v in values]

    fig = px.line_polar(r=[*values, values[0]], theta=[*labels, labels[0]], line_close=True,
                        title=f"<b>{title}</b>")
    fig.update_traces(fill='toself')
    fig.update_layout(height=420, polar=dict(
        radialaxis=dict(visible=True, range=[0, 1])))
    return fig


def plot_movies_by_year(movies_df):
    """
    Plot a vertical bar chart of movie counts per year (interactive Plotly).
    Returns None when no usable year data.
    """
    if movies_df is None or movies_df.empty or 'year' not in movies_df.columns:
        return None

    df = movies_df.copy()
    df = df.dropna(subset=['year'])
    if df.empty:
        return None

    # Use integer years and aggregate counts
    df['year'] = df['year'].astype(int)
    counts = df.groupby('year').size().reset_index(name='count')
    counts = counts.sort_values('year')

    fig = px.bar(counts, x='year', y='count',
                 title='<b>Movie Count by Year</b>',
                 labels={'year': 'Year', 'count': 'Number of Movies'},
                 orientation='v',
                 height=700)
    fig.update_layout(xaxis={'type': 'category'},
                      bargap=0.2, template='plotly_white')
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

    # Explode genres to have one genre per row (do this on a copy)
    data = data.copy()
    data['genre'] = data['genres'].str.split('|')
    data = data.explode('genre')

    fig = px.violin(data, x='genre', y='rating',
                    title=title,
                    labels={'genre': 'Genre', 'rating': 'Rating'},
                    color='genre')
    return fig
