import os
import requests
import pandas as pd


# --- 1. CONFIGURATION & API KEYS ---
# API Key to fetch movie posters from The Movie Database (TMDB)
TMDB_API_KEY = "41f96d2fff5a05d9a8190729bc44d34b"

# --- 2. PROJECT DIRECTORIES ---
# Automatically get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
ASSETS_DIR = os.path.join(PROJECT_ROOT, 'assets')

# Create directories automatically if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)


def ensure_directories():
    """
    Ensure that all necessary directories for the project exist.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(ASSETS_DIR, exist_ok=True)


# --- 3. MODEL FILE PATHS ---
SVD_MODEL_PATH = os.path.join(MODELS_DIR, 'svd_model.pkl')
DL_MODEL_PATH = os.path.join(MODELS_DIR, 'dl_model.h5')
TFIDF_MATRIX_PATH = os.path.join(MODELS_DIR, 'tfidf_matrix.pkl')
MAPPINGS_PATH = os.path.join(MODELS_DIR, 'mappings.pkl')
POPULARITY_MODEL_PATH = os.path.join(MODELS_DIR, 'popularity_model.pkl')

# --- 4. PROCESSED DATA PATHS ---
# Using pickle for faster I/O with pandas
PROCESSED_MOVIES_PATH = os.path.join(DATA_DIR, 'movies_processed.pkl')
PROCESSED_RATINGS_PATH = os.path.join(DATA_DIR, 'ratings_processed.pkl')

# --- 5. HELPER FUNCTIONS ---


def fetch_poster(tmdb_id):
    """
    Fetches the movie poster URL from the TMDB API.
    """
    if str(tmdb_id) == 'nan' or not tmdb_id:
        return "https://via.placeholder.com/300x450?text=No+Image"

    try:
        clean_id = int(float(tmdb_id))
    except (ValueError, TypeError):
        return "https://via.placeholder.com/300x450?text=Invalid+ID"

    url = f"https://api.themoviedb.org/3/movie/{clean_id}?api_key={TMDB_API_KEY}"

    try:
        response = requests.get(url, timeout=2)
        if response.status_code == 200:
            data = response.json()
            if data.get('poster_path'):
                return "https://image.tmdb.org/t/p/w500" + data['poster_path']
    except requests.exceptions.RequestException:
        pass  # API call failed, will return default poster below

    return "https://via.placeholder.com/300x450?text=No+Poster"


def format_runtime(minutes):
    """
    Formats runtime from minutes to a 'Xh Ym' string.
    """
    if not minutes or pd.isna(minutes):
        return "N/A"
    try:
        m = int(float(minutes))
        return f"{m//60}h {m % 60}m"
    except (ValueError, TypeError):
        return "N/A"


def save_fig(fig, name):
    """
    Saves a Plotly or Matplotlib figure to the assets directory.
    Plotly figs are saved as interactive HTML, Matplotlib figs as static PNG.
    """
    if fig is None:
        print(f"⚠️ Could not generate visualization for {name}.")
        return

    # Check for Plotly figures
    if hasattr(fig, 'write_html'):
        fig.write_html(os.path.join(ASSETS_DIR, f"{name}.html"))
        print(f"🖼️ Saved interactive '{name}.html'")
    # Check for Matplotlib figures (like WordCloud)
    elif hasattr(fig, 'savefig'):
        fig.savefig(os.path.join(ASSETS_DIR, f"{name}.png"))
        print(f"🖼️ Saved static '{name}.png'")
