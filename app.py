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
                       PROCESSED_RATINGS_PATH)
from src.recommenders.content_based import ContentBasedRecommender
from src.recommenders.collaborative import CollaborativeRecommender
from src.recommenders.deep_learning import DeepLearningRecommender
from src.recommenders.hybrid import HybridRecommender
from src.recommenders.popularity import PopularityRecommender
from src.recommenders.mood_recommender import MoodRecommender
from src.data_loader import load_genome_data, load_tags_data
from src.analysis import (plot_rating_distribution, plot_activity_over_time,
                          plot_genre_treemap, plot_wordcloud,
                          plot_popularity_vs_quality, plot_genre_violin)

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

st.set_page_config(layout="wide", page_title="Cinema AI Pro", page_icon="🎬")

# --- 1. LOAD RESOURCES ---


def display_recs(recs_df):
    if recs_df is None or recs_df.empty:
        st.warning("No recommendations found.")
        return
    cols = st.columns(min(len(recs_df), 5))
    for idx, row in recs_df.head(5).iterrows():
        with cols[idx]:
            st.image(fetch_poster(row['tmdbId']), use_container_width=True)
            st.caption(f"**{row['title']}**")


@st.cache_resource
def load_system():
    """
    Load all necessary data and pre-trained models.
    Handles error checking if models are not found.
    """
    if not os.path.exists(PROCESSED_MOVIES_PATH):
        return None, "Models not found. Please run the training pipeline first."

    # Load data
    movies = pd.read_pickle(PROCESSED_MOVIES_PATH)
    ratings = pd.read_pickle(PROCESSED_RATINGS_PATH)
    genome_scores, genome_tags = load_genome_data()
    tags = load_tags_data()

    # Load models
    cb_model = ContentBasedRecommender.load(movies)
    cf_model = CollaborativeRecommender()
    cf_model.load()
    dl_model = DeepLearningRecommender()
    dl_model.load()
    pop_model = PopularityRecommender()
    pop_model.load()
    mood_model = MoodRecommender(genome_scores, genome_tags, movies)
    hb_model = HybridRecommender(cb_model, cf_model)

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
page = st.sidebar.radio("Navigation", [
    "📊 Dashboard", "🤖 AI Recommenders", "👤 User Center", "📈 Model Metrics"
])

# Shared UI elements
st.sidebar.markdown("---")
selected_user_id = st.sidebar.selectbox(
    "Select User ID (for demo)", app_data['ratings']['userId'].unique()[:30]
)

# --- 3. PAGE: DASHBOARD ---
if page == "📊 Dashboard":
    st.title("📊 Data & Insights Dashboard")
    st.markdown("An interactive exploration of the MovieLens dataset. "
                "Global charts are pre-generated for speed, while personalized charts are generated on-the-fly.")
    st.info(
        f"Personalized charts below are currently showing for **User #{selected_user_id}**. You can change the user in the sidebar.")

    # --- Load pre-generated visualizations for speed ---
    def load_html(filename):
        filepath = os.path.join("assets", filename)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        st.warning(
            f"Chart '{filename}' not found. Please run the training pipeline.")
        return None

    # Display all 6 visualizations
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("##### Rating Distribution")
        html = load_html("rating_distribution.html")
        if html:
            st.components.v1.html(html, height=400)

        st.markdown("##### Activity Over Time")
        html = load_html("activity_over_time.html")
        if html:
            st.components.v1.html(html, height=400)
    with c2:
        st.markdown("##### Genre Hierarchy (Treemap)")
        html = load_html("genre_treemap.html")
        if html:
            st.components.v1.html(html, height=400)

        st.markdown("##### Popularity vs. Quality")
        html = load_html("popularity_vs_quality.html")
        if html:
            st.components.v1.html(html, height=400)
    with c3:
        st.markdown("##### Rating Distribution by Genre")
        html = load_html("genre_violin.html")
        if html:
            st.components.v1.html(html, height=400)

        st.markdown("##### Popular User Tags")
        wordcloud_path = os.path.join("assets", "tags_wordcloud.png")
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
            display_recs(recs)

    # --- Content-Based Demo ---
    elif rec_type == "Content-Based":
        st.subheader("🎬 Recommend Movies Similar to...")
        movie_title = st.selectbox("Select a movie you like:",
                                   app_data['movies']['title'].unique())
        if st.button("Recommend", key="cb"):
            recs = app_models['content_based'].recommend(movie_title, top_k=5)
            display_recs(recs)

    # --- Collaborative Filtering Demo ---
    elif rec_type == "Collaborative Filtering":
        st.subheader(f"✨ Top Recommendations for User {selected_user_id}")
        if st.button("Recommend", key="cf"):
            recs = app_models['collaborative'].get_top_n_for_user(
                selected_user_id, app_data['movies'], n=5)
            display_recs(recs)

    # --- Hybrid Demo ---
    elif rec_type == "Hybrid (Content + Collaborative)":
        st.subheader("🤝 Hybrid Recommendations")
        movie_title = st.selectbox(
            "Select a movie you like:", app_data['movies']['title'].unique())
        if st.button("Recommend", key="hb"):
            recs = app_models['hybrid'].recommend(
                selected_user_id, movie_title, top_k=5)
            display_recs(recs)

    # --- Deep Learning Demo ---
    elif rec_type == "Deep Learning Ranking":
        st.subheader("🧠 Deep Learning Enhanced Recommendations")
        movie_title = st.selectbox(
            "Select a movie you like (seed for candidates):", app_data['movies']['title'].unique())
        if st.button("Recommend", key="dl"):
            # Get candidates from a fast model (e.g., content-based)
            candidates = app_models['content_based'].recommend(
                movie_title, top_k=50)
            if not candidates.empty:
                # Re-rank with DL model
                candidates['score'] = candidates['movieId'].apply(
                    lambda mid: app_models['deep_learning'].predict(selected_user_id, mid))
                recs = candidates.sort_values(
                    'score', ascending=False).head(5)
                display_recs(recs)
            else:
                st.warning("Could not generate candidates.")

    # --- Mood-Based Demo ---
    elif rec_type == "Mood-Based":
        st.subheader("😊 Movie Recommendations by Mood")
        mood = st.text_input(
            "How are you feeling? (e.g., funny, dark, romantic)")
        if st.button("Recommend", key="mood") and mood:
            recs = app_models['mood'].recommend(mood, top_k=5)
            display_recs(recs)


# --- 5. PAGE: USER CENTER ---
elif page == "👤 User Center":
    st.title(f"👤 User Center for #{selected_user_id}")

    # Display user's rating history
    st.subheader("Your Rating History")
    user_history = app_data['ratings'][app_data['ratings']
                                       ['userId'] == selected_user_id]
    user_history = pd.merge(
        user_history, app_data['movies'], on='movieId', how='left')
    st.dataframe(user_history[['title', 'rating', 'genres']].head(10))

    # Display personalized recommendations
    st.subheader("Personalized Recommendations For You")
    if st.button("Get My Recommendations", type="primary"):
        recs = app_models['collaborative'].get_top_n_for_user(
            selected_user_id, app_data['movies'], n=5)
        display_recs(recs)


# --- 6. PAGE: MODEL METRICS ---
elif page == "📈 Model Metrics":
    st.title("📈 Model Performance Metrics")
    st.markdown(
        "Evaluating the SVD model from the Surprise library, as it provides clear metrics.")

    metrics = app_models['collaborative'].test_metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RMSE", f"{metrics.get('RMSE', 0):.4f}")
    c2.metric("MAE", f"{metrics.get('MAE', 0):.4f}")
    c3.metric("Precision@10", f"{metrics.get('Precision@10', 0):.2%}")
    c4.metric("Recall@10", f"{metrics.get('Recall@10', 0):.2%}")

    st.info("""
    - **RMSE (Root Mean Squared Error):** Measures the average magnitude of the errors. Lower is better.
    - **MAE (Mean Absolute Error):** Similar to RMSE, but less sensitive to large errors. Lower is better.
    - **Precision@10:** Out of the top 10 movies we recommended, what proportion did the user actually like? Higher is better.
    - **Recall@10:** Out of all the movies the user liked, what proportion did we capture in our top 10 recommendations? Higher is better.
    """)
