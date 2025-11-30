from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
import pickle
from src.utils import TFIDF_MATRIX_PATH


class ContentBasedRecommender:
    def __init__(self, movies, tfidf_matrix=None):
        self.movies = movies.reset_index(drop=True)
        self.tfidf_matrix = tfidf_matrix

    def save(self):
        with open(TFIDF_MATRIX_PATH, 'wb') as f:
            pickle.dump(self.tfidf_matrix, f)

    @classmethod
    def load(cls, movies):
        with open(TFIDF_MATRIX_PATH, 'rb') as f:
            matrix = pickle.load(f)
        return cls(movies, matrix)

    def recommend(self, movie_title, top_k=20):
        # Guard: if TF-IDF matrix or movies are missing, return empty
        if self.tfidf_matrix is None or self.movies is None or self.movies.empty:
            return pd.DataFrame()

        # Build mapping from title -> positional index (0..n-1)
        title_to_pos = {title: pos for pos, title in enumerate(
            self.movies['title'].astype(str).tolist())}

        # Exact match required — if not found, return empty
        if movie_title not in title_to_pos:
            return pd.DataFrame()

        idx = int(title_to_pos[movie_title])

        # Compute cosine similarities safely
        try:
            cosine_sim = linear_kernel(
                self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
        except Exception:
            # If the tfidf matrix doesn't support indexing by this idx, fail gracefully
            return pd.DataFrame()

        sim_scores = list(enumerate(cosine_sim))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[
            1:top_k + 1]

        # Keep only indices that are within bounds
        valid_movie_indices = [
            i for i, _ in sim_scores if 0 <= i < len(self.movies)]
        if not valid_movie_indices:
            return pd.DataFrame()

        # Preserve recommendation ordering
        try:
            result = self.movies.reset_index(
                drop=True).iloc[valid_movie_indices]
        except Exception:
            result = self.movies[self.movies.index.isin(valid_movie_indices)]

        return result
