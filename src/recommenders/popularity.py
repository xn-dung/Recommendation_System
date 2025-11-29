"""
This module defines a Popularity-Based Recommender.

This model recommends movies based on their overall popularity, calculated 
from the average rating and the number of votes. It's a simple yet effective
baseline, especially for new users with no interaction history.
"""
import pandas as pd
import pickle
from src.utils import POPULARITY_MODEL_PATH


class PopularityRecommender:
    def __init__(self, movies_df=None, ratings_df=None):
        self.movies = movies_df
        self.ratings = ratings_df
        self.model = None

    def train(self):
        """
        Trains the popularity model by calculating a weighted score for each movie.
        The score is a simple weighted average of the mean rating and the number of votes.
        """
        if self.ratings is None or self.movies is None:
            raise ValueError(
                "Ratings and movies data must be provided for training.")

        movie_stats = self.ratings.groupby('movieId').agg(
            {'rating': ['mean', 'count']}).reset_index()
        movie_stats.columns = ['movieId', 'avg_rating', 'vote_count']

        # A simple weighted rating formula
        C = movie_stats['avg_rating'].mean()
        m = movie_stats['vote_count'].quantile(
            0.8)  # Minimum votes to be listed

        def weighted_rating(x, m=m, C=C):
            v = x['vote_count']
            R = x['avg_rating']
            return (v / (v + m) * R) + (m / (v + m) * C)

        movie_stats = movie_stats[movie_stats['vote_count'] >= m]
        movie_stats['score'] = movie_stats.apply(weighted_rating, axis=1)

        # Merge with movie titles
        self.model = pd.merge(
            movie_stats, self.movies, on='movieId', how='left')
        self.model = self.model.sort_values(
            'score', ascending=False).reset_index(drop=True)

    def recommend(self, top_k=10):
        """
        Returns the top K most popular movies.
        """
        if self.model is None:
            raise RuntimeError(
                "Model has not been trained. Please call train() first.")
        return self.model.head(top_k)

    def save(self):
        """
        Saves the trained popularity model to a pickle file.
        """
        if self.model is None:
            raise RuntimeError("Cannot save a model that hasn't been trained.")
        with open(POPULARITY_MODEL_PATH, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self):
        """
        Loads the popularity model from a pickle file.
        """
        with open(POPULARITY_MODEL_PATH, 'rb') as f:
            self.model = pickle.load(f)
