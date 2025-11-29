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
        # Tìm phim theo tên (case insensitive)
        mask = self.movies['title'].str.contains(
            movie_title, case=False, regex=False)
        if not mask.any():
            return pd.DataFrame()

        idx = mask.idxmax()
        cosine_sim = linear_kernel(
            self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
        sim_scores = list(enumerate(cosine_sim))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[
            1:top_k+1]

        movie_indices = [i[0] for i in sim_scores]
        return self.movies.iloc[movie_indices]
