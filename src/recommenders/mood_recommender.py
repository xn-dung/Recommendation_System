import pandas as pd


class MoodRecommender:
    def __init__(self, genome_scores_df, genome_tags_df, movies_df):
        self.genome_scores = genome_scores_df
        self.genome_tags = genome_tags_df
        self.movies = movies_df

    def recommend(self, mood_keyword, min_relevance=0.8, top_k=5):
        """
        Finds movies that highly match a specific mood tag.
        """
        if not mood_keyword or not isinstance(mood_keyword, str):
            return pd.DataFrame()

        # Case-insensitive search for the mood keyword in tags
        tag_row = self.genome_tags[self.genome_tags['tag'].str.contains(
            mood_keyword, case=False, na=False)]

        if tag_row.empty:
            return pd.DataFrame()

        tag_id = tag_row.iloc[0]['tagId']

        relevant_movies = self.genome_scores[
            (self.genome_scores['tagId'] == tag_id) &
            (self.genome_scores['relevance'] > min_relevance)
        ]

        top_movies = relevant_movies.sort_values(
            'relevance', ascending=False).head(top_k)
        result = pd.merge(top_movies, self.movies, on='movieId')

        return result[['movieId', 'title', 'genres', 'relevance', 'tmdbId']]
