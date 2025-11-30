"""
This module creates a professional and interactive Streamlit web application
for the Movie Recommendation System.

The app features a multi-page layout with:
- An interactive dashboard for data exploration.
- A demo page to test various AI recommender models.
- A personalized user history and recommendation page.
- A model evaluation and metrics overview.
"""
import streamlit as st
import pandas as pd
import sys
import os
from src.utils import (fetch_poster, PROCESSED_MOVIES_PATH,
                       PROCESSED_RATINGS_PATH, ASSETS_DIR)
from src.data_fetcher import ensure_file_from_env
from src.recommenders.content_based import ContentBasedRecommender
from src.recommenders.collaborative import CollaborativeRecommender
from src.recommenders.deep_learning import DeepLearningRecommender
from src.recommenders.hybrid import HybridRecommender
from src.recommenders.popularity import PopularityRecommender
from src.recommenders.mood_recommender import MoodRecommender
from src.data_loader import load_genome_data, load_tags_data
from src.analysis import (plot_rating_distribution, plot_activity_over_time,
                          plot_genre_treemap, plot_wordcloud, plot_wordcloud_by_year,
                          plot_popularity_vs_quality, plot_genre_violin,
                          plot_movies_by_year, plot_metrics_summary, plot_metrics_radar)
import time

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

st.set_page_config(layout="wide", page_title="Cinema AI Pro", page_icon="🎬")

# --- 1. LOAD RESOURCES ---


def display_recs(recs_df):
    if recs_df is None or recs_df.empty:
        st.info("No films found.")
        return
    # Show all rows in a grid — 5 items per row (wrap as needed)
    max_per_row = 5
    rows = recs_df.reset_index(drop=True)
    for start in range(0, len(rows), max_per_row):
        chunk = rows.iloc[start:start + max_per_row]
        cols = st.columns(min(len(chunk), max_per_row))
        for col_idx, (_, row) in enumerate(chunk.iterrows()):
            with cols[col_idx]:
                # Safely fetch poster id and year; use placeholders if missing
                tmdb_id = row.get('tmdbId') if isinstance(
                    row, pd.Series) else None
                title = row.get('title') if isinstance(
                    row, pd.Series) else None
                year = row.get('year') if isinstance(row, pd.Series) else None

                if tmdb_id:
                    try:
                        st.image(fetch_poster(tmdb_id),
                                 use_container_width=True)
                    except Exception:
                        st.text("[Poster unavailable]")
                else:
                    st.text("[Poster unavailable]")

                caption = "Unknown title" if not title else str(title)
                if year:
                    caption = f"**{caption}** ({year})"
                else:
                    caption = f"**{caption}**"

                st.caption(caption)


@st.cache_resource
def load_system():
    """
    Load all necessary data and pre-trained models.
    Handles error checking if models are not found.
    """
    if not os.path.exists(PROCESSED_MOVIES_PATH):
        # Try to automatically download processed datasets from environment/Streamlit secrets.
        # Set PROCESSED_MOVIES_URL and PROCESSED_RATINGS_URL when deploying (Streamlit secrets or env).
        # Example Streamlit secret keys: {'PROCESSED_MOVIES_URL': 'https://.../movies_processed.pkl', 'PROCESSED_RATINGS_URL': 'https://.../ratings_processed.pkl'}
        got_movies = ensure_file_from_env(PROCESSED_MOVIES_PATH, ('PROCESSED_MOVIES_URL', 'PROCESSED_DATA_URL'))
        got_ratings = True
        if not os.path.exists(PROCESSED_RATINGS_PATH):
            got_ratings = ensure_file_from_env(PROCESSED_RATINGS_PATH, ('PROCESSED_RATINGS_URL', 'PROCESSED_DATA_URL'))

        if not got_movies or not os.path.exists(PROCESSED_MOVIES_PATH):
            st.error("Processed data not found. Please run the training pipeline first or provide download URLs via environment/Streamlit secrets.")
            st.stop()

        # If we successfully downloaded movies but not ratings, try to give clearer message
        if not os.path.exists(PROCESSED_RATINGS_PATH):
            st.error("Processed ratings file not found. Provide PROCESSED_RATINGS_URL (env/secret) or run the training pipeline.")
            st.stop()

    # Load data
    movies = pd.read_pickle(PROCESSED_MOVIES_PATH)
    ratings = pd.read_pickle(PROCESSED_RATINGS_PATH)
    genome_scores, genome_tags = load_genome_data()
    tags = load_tags_data()

    # Load models
    try:
        cb_model = ContentBasedRecommender.load(movies)
        cf_model = CollaborativeRecommender(ratings)
        cf_model.load()
        dl_model = DeepLearningRecommender()
        dl_model.load()
        pop_model = PopularityRecommender()
        pop_model.load()
        mood_model = MoodRecommender(genome_scores, genome_tags, movies)
        hb_model = HybridRecommender(cb_model, cf_model)
    except FileNotFoundError:
        st.error(
            "One or more model files are missing. Please run the training pipeline.")
        st.stop()

    models = {
        "content_based": cb_model, "collaborative": cf_model,
        "deep_learning": dl_model, "hybrid": hb_model,
        "popularity": pop_model, "mood": mood_model
    }
    data = {"movies": movies, "ratings": ratings, "tags": tags}
    return data, models, None


with st.spinner("Initializing AI System... Please wait."):
    app_data, app_models, error = load_system()

if error:
    st.error(f"⚠️ {error}")
    st.stop()

# --- 2. SIDEBAR NAVIGATION ---
st.sidebar.title("🎬 Cinema AI Pro")

# Year filter — only show and apply it when 'year' exists and is numeric
has_year = 'year' in app_data['movies'].columns and pd.api.types.is_numeric_dtype(
    app_data['movies']['year'])
if has_year:
    min_year = int(app_data['movies']['year'].min())
    max_year = int(app_data['movies']['year'].max())
    selected_year_range = st.sidebar.slider(
        "Filter by Year:",
        min_year, max_year, (min_year, max_year)
    )

    # Filter data based on year range
    movies_filtered = app_data['movies'][
        (app_data['movies']['year'] >= selected_year_range[0]) &
        (app_data['movies']['year'] <= selected_year_range[1])
    ]
    ratings_filtered = app_data['ratings'][
        app_data['ratings']['movieId'].isin(movies_filtered['movieId'])
    ]
else:
    # No usable 'year' column — show full datasets and a helpful note
    st.sidebar.info(
        "Year filter disabled — movie dataset has no numeric 'year' column.")
    movies_filtered = app_data['movies']
    ratings_filtered = app_data['ratings']
    selected_year_range = None
app_data_filtered = {"movies": movies_filtered, "ratings": ratings_filtered}

# Navigation control (was missing, causing 'page' to be undefined)
page = st.sidebar.radio("Navigation", [
    "📊 Dashboard", "🤖 AI Recommenders", "👤 User Center", "📈 Model Metrics"
])

st.sidebar.markdown("---")
# Choose a demo user — include the full set of userIds available in the processed ratings
# (we allow full selection; re-run pipeline with KEEP_ALL_USERS to add small/rare IDs)
all_user_ids = sorted(app_data['ratings']['userId'].unique())
# If the dataset has a reasonable number of users, show full dropdown (searchable).
# Otherwise provide a quick-pick plus a validated manual entry to keep the UI responsive.
USER_DROPDOWN_LIMIT = 3000
if len(all_user_ids) <= USER_DROPDOWN_LIMIT:
    selected_user_id = st.sidebar.selectbox(
        "Select User ID (for demo)", all_user_ids)
else:
    st.sidebar.info(
        f"Dataset contains {len(all_user_ids):,} users — using quick-pick + validated entry to stay responsive.")
    quick_limit = 1000
    quick_options = all_user_ids[:quick_limit]
    quick_choice = st.sidebar.selectbox(
        "Quick pick (first 1k users)", quick_options)

    manual_input = st.sidebar.text_input(
        "Or enter User ID (must exist in dataset)")
    selected_user_id = quick_choice
    if manual_input.strip():
        try:
            manual_id = int(manual_input.strip())
            if manual_id in all_user_ids:
                selected_user_id = manual_id
            else:
                st.sidebar.error(
                    "Entered userId not found in dataset — please enter a valid userId from ratings.csv")
        except ValueError:
            st.sidebar.error("User ID must be an integer value.")

if ratings_filtered.empty:
    st.sidebar.warning(
        "No user ratings found for the selected year range — using full dataset")

# Note: only dropdown selection is allowed for user IDs (no manual input) — the processed
# dataset may have filtered out many low-activity users. If you need a specific small ID
# like 1 or 2, re-run the pipeline with KEEP_ALL_USERS=1 and appropriate SAMPLE_SIZE/USE_FULL_RATINGS.

# --- 3. PAGE: DASHBOARD ---
if page == "📊 Dashboard":
    st.title("📊 Data & Insights Dashboard")
    st.markdown("An interactive exploration of the MovieLens dataset. "
                "Global charts are pre-generated for speed, while personalized charts are generated on-the-fly.")
    st.info(
        f"Personalized charts below are currently showing for **User #{selected_user_id}**. You can change the user in the sidebar.")

    # --- Load pre-generated visualizations for speed ---
    def load_html(filename):
        """
        Robustly locate and return the contents of an HTML asset.

        Checks several likely locations in order:
        1. ASSETS_DIR (canonical absolute directory from src.utils)
        2. repo-relative <repo_root>/assets/ (covers running from other working dirs)
        3. ./assets/ relative to the current working directory
        4. raw filename (if caller passed an absolute path)

        Returns the file contents or None if not found.
        """
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..'))
        candidates = [
            os.path.join(ASSETS_DIR, filename),
            os.path.join(repo_root, 'assets', filename),
            os.path.join(os.getcwd(), 'assets', filename),
            filename,
        ]

        for filepath in candidates:
            if filepath and os.path.exists(filepath):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        return f.read()
                except Exception:
                    # read failed — try next candidate
                    continue

        st.warning(
            f"Chart '{filename}' not found in assets. Run training pipeline (python train_pipeline.py) or check assets folder.")
        return None

    # Display larger, vertical interactive visualizations for laptop screens
    st.markdown("##### Rating Distribution")
    # try to load pre-generated HTML, otherwise render from data
    html = load_html("rating_distribution.html")
    if html:
        st.components.v1.html(html, height=680)
    else:
        fig = plot_rating_distribution(
            app_data_filtered['ratings'], user_id=selected_user_id)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Activity Over Time")
    html = load_html("activity_over_time.html")
    if html:
        st.components.v1.html(html, height=680)
    else:
        fig = plot_activity_over_time(
            app_data_filtered['ratings'], user_id=selected_user_id)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Movie Count by Year")
    # Prefer pre-generated HTML for speed and consistency; fallback to
    # rendering on-the-fly if it isn't present.
    html = load_html("movies_by_year.html")
    if html:
        st.components.v1.html(html, height=680)
    else:
        # Render a vertical Plotly bar chart (interactive) using movies_filtered
        fig = plot_movies_by_year(app_data_filtered['movies'])
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Rating Distribution by Genre")
    html = load_html("genre_violin.html")
    if html:
        st.components.v1.html(html, height=680)
    else:
        fig = plot_genre_violin(
            app_data_filtered['movies'], app_data_filtered['ratings'], user_id=selected_user_id)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True, height=680)

    st.markdown("##### Popular User Tags")
    # Allow viewing tags by year (select a year or 'All') and animate through years
    if has_year:
        years = sorted(app_data['movies']['year'].dropna().astype(
            int).unique().tolist())
        year_options = ['All'] + years
        selected_tag_year = st.selectbox(
            "Tag Year (All = aggregate):", year_options, index=0)

        # Animate button
        if st.button("Animate tags over years"):
            placeholder = st.empty()
            for yr in years:
                # Prefer pre-generated asset
                html_name = f"tags_wordcloud_{yr}.html"
                html = load_html(html_name)
                if html:
                    placeholder.components.v1.html(html, height=480)
                else:
                    fig = plot_wordcloud_by_year(
                        app_data['tags'], app_data['movies'], yr)
                    if fig is not None:
                        placeholder.pyplot(fig)
                    else:
                        placeholder.write(f"No tags for {yr}")
                time.sleep(0.8)

        # Show selected year
        if selected_tag_year == 'All':
            html = load_html("tags_wordcloud.html")
            if html:
                st.components.v1.html(html, height=480)
            else:
                wordcloud_path = os.path.join(ASSETS_DIR, "tags_wordcloud.png")
                if os.path.exists(wordcloud_path):
                    st.image(wordcloud_path, use_container_width=True)
                else:
                    st.warning("WordCloud not found.")
        else:
            # try pre-generated per-year html then fallback to on-the-fly plot
            html = load_html(f"tags_wordcloud_{selected_tag_year}.html")
            if html:
                st.components.v1.html(html, height=480)
            else:
                fig = plot_wordcloud_by_year(
                    app_data['tags'], app_data['movies'], selected_tag_year)
                if fig is not None:
                    st.pyplot(fig)
                else:
                    st.info("No tags found for selected year.")
    else:
        html = load_html("tags_wordcloud.html")
        if html:
            st.components.v1.html(html, height=680)
        else:
            wordcloud_path = os.path.join(ASSETS_DIR, "tags_wordcloud.png")
            if os.path.exists(wordcloud_path):
                st.image(wordcloud_path, use_container_width=True)
            else:
                st.warning("WordCloud not found.")


# --- 4. PAGE: AI RECOMMENDERS ---
elif page == "🤖 AI Recommenders":
    st.title("🤖 AI Recommender Demos")
    rec_type = st.selectbox("Choose a Recommender Model",
                            ["Popularity", "Content-Based", "Collaborative Filtering",
                             "Hybrid (Content + Collaborative)", "Deep Learning Ranking", "Mood-Based"])

    # --- Popularity Demo ---
    if rec_type == "Popularity":
        st.subheader("🏆 Top 10 Most Popular Movies")
        if st.button("Recommend", key="pop"):
            recs = app_models['popularity'].recommend(top_k=10)
            # Ensure popularity suggestions respect the selected year range
            if selected_year_range is not None and not recs.empty:
                recs = recs[(recs['year'] >= selected_year_range[0])
                            & (recs['year'] <= selected_year_range[1])]
            display_recs(recs)

    # --- Content-Based Demo ---
    elif rec_type == "Content-Based":
        st.subheader("🎬 Recommend Movies Similar to...")
        # Allow selecting any movie as the seed (not limited by the current year filter)
        titles = app_data['movies']['title'].unique()
        if len(titles) == 0:
            st.info(
                "No movies available for the selected year range — change the year filter to get recommendations.")
        else:
            movie_title = st.selectbox("Select a movie you like:", titles)
            if st.button("Recommend", key="cb"):
                # Request a larger candidate pool then apply the currently-selected year filter
                recs = app_models['content_based'].recommend(
                    movie_title, top_k=100)
                if selected_year_range is not None and not recs.empty:
                    recs = recs[(recs['year'] >= selected_year_range[0]) & (
                        recs['year'] <= selected_year_range[1])]
                if recs is None or recs.empty:
                    st.info("No similar movies found for the selected title.")
                else:
                    display_recs(recs)

    # --- Collaborative Filtering Demo ---
    elif rec_type == "Collaborative Filtering":
        st.subheader(f"✨ Top Recommendations for User {selected_user_id}")
        if st.button("Recommend", key="cf"):
            # Return a larger set of candidate recommendations (if available) and already constrained by filtered movies
            recs = app_models['collaborative'].get_top_n_for_user(
                selected_user_id, app_data_filtered['movies'], n=50)
            display_recs(recs)

    # --- Hybrid Demo ---
    elif rec_type == "Hybrid (Content + Collaborative)":
        st.subheader("🤝 Hybrid Recommendations")
        # Allow selecting a seed movie from the full catalog
        titles = app_data['movies']['title'].unique()
        if len(titles) == 0:
            st.info(
                "No movies available for hybrid recommendations. Adjust year filter or run the training pipeline.")
        else:
            movie_title = st.selectbox("Select a movie you like:", titles)
            if st.button("Recommend", key="hb"):
                # Ask for a larger candidate pool and then filter to the selected year range
                recs = app_models['hybrid'].recommend(
                    selected_user_id, movie_title, top_k=100)
                if selected_year_range is not None and not recs.empty:
                    recs = recs[(recs['year'] >= selected_year_range[0]) & (
                        recs['year'] <= selected_year_range[1])]
                if recs is None or recs.empty:
                    st.info(
                        "No hybrid recommendations available for this title/user.")
                else:
                    display_recs(recs)

    # --- Deep Learning Demo ---
    elif rec_type == "Deep Learning Ranking":
        st.subheader("🧠 Deep Learning Enhanced Recommendations")
        # Candidate seeds should be selected from the full catalog (not constrained by the year filter)
        titles = app_data['movies']['title'].unique()
        if len(titles) == 0:
            st.info(
                "No movies available for deep-learning re-ranking. Adjust year filter or run the training pipeline.")
        else:
            movie_title = st.selectbox(
                "Select a movie you like (seed for candidates):", titles)
            if st.button("Recommend", key="dl"):
                # Get candidates from a fast model (e.g., content-based)
                candidates = app_models['content_based'].recommend(
                    movie_title, top_k=200)
                if not candidates.empty:
                    # apply year filter to the candidate set if requested
                    if selected_year_range is not None:
                        candidates = candidates[(candidates['year'] >= selected_year_range[0]) & (
                            candidates['year'] <= selected_year_range[1])]
                    if candidates.empty:
                        st.info(
                            "No candidates remain after applying the year filter.")
                        # nothing to display — skip DL reranking
                        candidates = pd.DataFrame()
                    # Re-rank with DL model

                    def safe_predict(mid):
                        try:
                            return float(app_models['deep_learning'].predict(selected_user_id, mid))
                        except Exception:
                            # If something goes wrong with the DL model, return a neutral score
                            return 0.0

                    candidates['score'] = candidates['movieId'].apply(
                        safe_predict)
                    recs = candidates.sort_values(
                        'score', ascending=False).head(50)
                    display_recs(recs)
                else:
                    st.warning("Could not generate candidates.")

    # --- Mood-Based Demo ---
    elif rec_type == "Mood-Based":
        st.subheader("😊 Movie Recommendations by Mood")
        mood = st.text_input(
            "How are you feeling? (e.g., funny, dark, romantic)")
        if st.button("Recommend", key="mood") and mood:
            recs = app_models['mood'].recommend(mood, top_k=100)
            if selected_year_range is not None and not recs.empty:
                recs = recs[(recs['year'] >= selected_year_range[0])
                            & (recs['year'] <= selected_year_range[1])]
            display_recs(recs)


# --- 5. PAGE: USER CENTER ---
elif page == "👤 User Center":
    st.title(f"👤 User Center for #{selected_user_id}")

    # Display user's rating history
    st.subheader("Your Rating History")
    user_history = app_data_filtered['ratings'][app_data_filtered['ratings']
                                                ['userId'] == selected_user_id]
    if user_history is None or user_history.empty:
        st.info("No rating history found for this user in the selected year range.")
    else:
        user_history = pd.merge(
            user_history, app_data_filtered['movies'], on='movieId', how='left')
        st.dataframe(user_history[['title', 'rating', 'genres']].head(10))

    # Display personalized recommendations
    st.subheader("Personalized Recommendations For You")
    if st.button("Get My Recommendations", type="primary"):
        recs = app_models['collaborative'].get_top_n_for_user(
            selected_user_id, app_data_filtered['movies'], n=50)
        if recs is None or getattr(recs, 'empty', False):
            st.info(
                "No recommendations available for this user in the selected year range.")
        else:
            display_recs(recs)


# --- 6. PAGE: MODEL METRICS ---
elif page == "📈 Model Metrics":
    st.title("📈 Model Performance Metrics")
    st.markdown(
        "Evaluating the SVD model from the Surprise library, as it provides clear metrics.")

    # Collect metrics from the trained collaborative model
    metrics = app_models['collaborative'].test_metrics or {}

    # A quick composite 'Quality Score' that mixes several metrics into a 0..100% score.
    rmse = float(metrics.get('RMSE', 0.0))
    # Convert RMSE into a normalized score [0,1] where lower RMSE -> closer to 1.
    # Rating scale is 0.5..5.0 so max error ~4.5; use that as a rough normalization.
    rmse_score = max(0.0, 1.0 - (rmse / 4.5))

    # metrics expected in 0..1 range
    ndcg = float(metrics.get('NDCG@10', 0.0))
    mapp = float(metrics.get('MAP@10', 0.0))
    precision = float(metrics.get('Precision@10', 0.0))
    recall = float(metrics.get('Recall@10', 0.0))
    hit = float(metrics.get('HitRate@10', 0.0))

    # A simple weighted blend into one 'quality' number to make the dashboard more accessible.
    # These weights are heuristic and easy to adjust.
    weights = {
        'rmse': 0.15, 'ndcg': 0.25, 'map': 0.2,
        'precision': 0.2, 'recall': 0.1, 'hit': 0.1
    }
    composite = (
        rmse_score * weights['rmse'] + ndcg * weights['ndcg'] + mapp * weights['map'] +
        precision * weights['precision'] + recall *
        weights['recall'] + hit * weights['hit']
    )

    # Top area: show a large composite score and a compact metrics table
    left, right = st.columns([2, 3])
    with left:
        st.metric(label="Model Quality Score", value=f"{composite*100:.1f}%",
                  delta=f"RMSE: {rmse:.3f}")
        st.write("\n")

    with right:
        # Nicely formatted metrics table
        df = pd.DataFrame.from_dict(metrics, orient='index', columns=['value'])
        df = df.reset_index().rename(columns={'index': 'metric'})
        # Format percentages where appropriate
        df['display'] = df.apply(lambda r: f"{r.value:.2%}" if 'Precision' in r.metric or 'Recall' in r.metric or 'HitRate' in r.metric else (
            f"{r.value:.4f}" if isinstance(r.value, float) else str(r.value)), axis=1)
        st.table(df[['metric', 'display']])

    st.markdown("---")

    # Visualizations: summary bar and radar plot
    fig_bar = plot_metrics_summary(metrics, title="Model Metrics (raw)")
    fig_radar = plot_metrics_radar({
        'NDCG@10': ndcg, 'MAP@10': mapp, 'Precision@10': precision,
        'Recall@10': recall, 'HitRate@10': hit, 'RMSE_score': rmse_score
    }, title="Normalized Model Overview")

    if fig_bar is not None:
        st.plotly_chart(fig_bar, use_container_width=True)
    if fig_radar is not None:
        st.plotly_chart(fig_radar, use_container_width=True)

    st.info("""
    Useful metrics:
    - **NDCG@10 / MAP@10:** Ranking-aware metrics that reward putting the most relevant items near the top.
    - **Precision/Recall@10:** Direct measures of top-K recommendation usefulness for users.
    - **HitRate@10:** Simple hit/no-hit measure that is easy to interpret.
    - **Coverage:** How many distinct items are shown in recommendations (diversity proxy).

    The `Model Quality Score` is a dashboard-friendly composite of several metrics (higher is better).
    """)
