"""
This module handles loading the MovieLens 25M dataset efficiently.
It includes caching mechanisms and data sampling to prevent memory overflow
on Streamlit Cloud deployment.
"""
import os
import pandas as pd
import streamlit as st
from src.utils import DATA_DIR


@st.cache_data
def load_data(ratings_nrows: int | None = 500000):
    """
    Loads main datasets (movies, ratings) with sampling to ensure performance.
    """
    # Compose full paths using DATA_DIR from src.utils so the loader
    # works reliably on local, Kaggle, and Colab environments.
    movies_path = os.path.join(DATA_DIR, 'movies.csv')
    links_path = os.path.join(DATA_DIR, 'links.csv')
    ratings_path = os.path.join(DATA_DIR, 'ratings.csv')

    # Basic checks and helpful errors if files are missing
    if not os.path.exists(movies_path):
        raise FileNotFoundError(
            f"movies.csv not found at {movies_path}. Please place the dataset in the data/ folder or set DATA_DIR accordingly.")
    if not os.path.exists(links_path):
        raise FileNotFoundError(
            f"links.csv not found at {links_path}. Please place the dataset in the data/ folder or set DATA_DIR accordingly.")
    if not os.path.exists(ratings_path):
        raise FileNotFoundError(
            f"ratings.csv not found at {ratings_path}. Please place the dataset in the data/ folder or set DATA_DIR accordingly.")

    movies = pd.read_csv(movies_path)
    links = pd.read_csv(links_path)
    movies = pd.merge(movies, links, on='movieId', how='left')
    # Limit rows to avoid memory issues in cloud environments
    # Allow callers to request a limited number of rows for memory-safety (default 500k)
    # or pass ratings_nrows=None to read the full file (useful for MovieLens-25M).
    ratings = pd.read_csv(ratings_path, nrows=ratings_nrows)
    return movies, ratings


@st.cache_data
def load_genome_data():
    """
    Loads the genome dataset (scores and tags).
    This data is used for mood-based recommendations and analysis.
    """
    scores_path = os.path.join(DATA_DIR, 'genome-scores.csv')
    tags_path = os.path.join(DATA_DIR, 'genome-tags.csv')

    if not os.path.exists(scores_path) or not os.path.exists(tags_path):
        raise FileNotFoundError(
            f"Genome files not found under {DATA_DIR}. Please ensure 'genome-scores.csv' and 'genome-tags.csv' are present.")

    genome_scores = pd.read_csv(scores_path, nrows=1000000)
    genome_tags = pd.read_csv(tags_path)
    return genome_scores, genome_tags


@st.cache_data
def load_tags_data():
    """
    Loads user-generated tags data.
    """
    tags_path = os.path.join(DATA_DIR, 'tags.csv')
    if not os.path.exists(tags_path):
        raise FileNotFoundError(
            f"tags.csv not found at {tags_path}. Please add it to the data directory.")
    tags = pd.read_csv(tags_path, nrows=50000)
    return tags
