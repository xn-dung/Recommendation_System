import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def preprocess_data(movies, ratings, sample_size=None, keep_all_users=True, min_movie_votes=20, min_user_ratings=20):
    """
    Preprocess raw MovieLens data.

    - movies: raw movies DataFrame (expects 'title', 'genres', 'movieId')
    - ratings: raw ratings DataFrame
    - sample_size: if provided (int), sample that many rating rows after filtering (for memory safety); None leaves data un-sampled.
    - keep_all_users: if True, do not filter out low-activity users or low-popularity movies. This preserves original user/movie ids so small user ids (1,2, ...) remain.
    - min_movie_votes, min_user_ratings: used when keep_all_users is False to filter sparsity.
    """

    # 1. Clean Text
    movies['genres'] = movies['genres'].fillna('')
    movies['genres_str'] = movies['genres'].str.replace('|', ' ')

    # 1b. Extract year from title when available (e.g., 'Toy Story (1995)')
    # Some datasets don't include an explicit release year column, so we parse it.
    def extract_year(title):
        try:
            if not isinstance(title, str):
                return None
            # year usually appears in parentheses at end
            import re
            m = re.search(r"\((\d{4})\)$", title.strip())
            if m:
                return int(m.group(1))
        except Exception:
            return None
        return None

    movies['year'] = movies['title'].apply(extract_year)

    # --- later we compute TF-IDF on the filtered movie set so the matrix
    # aligns with movies_filtered indices. (This prevents index mismatches
    # between titles and TF-IDF rows.)

    # 3. Filter Sparsity (optional)
    if keep_all_users:
        # Preserve all users and movies — do not apply sparsity filters.
        ratings_filtered = ratings.copy()
    else:
        movie_counts = ratings['movieId'].value_counts()
        popular_movies = movie_counts[movie_counts >= min_movie_votes].index

        user_counts = ratings['userId'].value_counts()
        active_users = user_counts[user_counts >= min_user_ratings].index

        ratings_filtered = ratings[
            (ratings['movieId'].isin(popular_movies)) &
            (ratings['userId'].isin(active_users))
        ]

    # 4. Optional Sampling (for memory-constrained training)
    if sample_size is not None and len(ratings_filtered) > sample_size:
        ratings_filtered = ratings_filtered.sample(
            n=sample_size, random_state=42)

    movies_filtered = movies[movies['movieId'].isin(
        ratings_filtered['movieId'].unique())].copy()

    # 2 (moved). TF-IDF on the filtered movies only so matrix rows match
    # movies_filtered row indices.
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_filtered['genres_str'])

    # 5. Encoding for Deep Learning
    user_ids = ratings_filtered['userId'].unique().tolist()
    user2user_encoded = {x: i for i, x in enumerate(user_ids)}

    movie_ids = ratings_filtered['movieId'].unique().tolist()
    movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}

    ratings_filtered['user_encoded'] = ratings_filtered['userId'].map(
        user2user_encoded)
    ratings_filtered['movie_encoded'] = ratings_filtered['movieId'].map(
        movie2movie_encoded)

    mappings = {
        'u2u': user2user_encoded,
        'm2m': movie2movie_encoded,
        'num_users': len(user_ids),
        'num_movies': len(movie_ids)
    }

    return movies_filtered, ratings_filtered, tfidf_matrix, mappings
