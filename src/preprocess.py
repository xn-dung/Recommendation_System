import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def preprocess_data(movies, ratings, sample_size=150000):
    # 1. Clean Text
    movies['genres'] = movies['genres'].fillna('')
    movies['genres_str'] = movies['genres'].str.replace('|', ' ')

    # 2. TF-IDF
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['genres_str'])

    # 3. Filter Sparsity
    movie_counts = ratings['movieId'].value_counts()
    popular_movies = movie_counts[movie_counts >= 20].index

    user_counts = ratings['userId'].value_counts()
    active_users = user_counts[user_counts >= 20].index

    ratings_filtered = ratings[
        (ratings['movieId'].isin(popular_movies)) &
        (ratings['userId'].isin(active_users))
    ]

    # 4. Sampling
    if len(ratings_filtered) > sample_size:
        ratings_filtered = ratings_filtered.sample(
            n=sample_size, random_state=42)

    movies_filtered = movies[movies['movieId'].isin(
        ratings_filtered['movieId'].unique())].copy()

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
