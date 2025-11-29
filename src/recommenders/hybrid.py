import pandas as pd


class HybridRecommender:
    def __init__(self, content_model, svd_model):
        self.content_model = content_model
        self.svd_model = svd_model

    def recommend(self, user_id, movie_title, top_k=10):
        content_recs = self.content_model.recommend(movie_title, top_k=50)

        if content_recs.empty:
            return pd.DataFrame()

        preds = []
        for mid in content_recs['movieId']:
            preds.append(self.svd_model.predict(user_id, mid))

        content_recs = content_recs.copy()
        content_recs['predicted_rating'] = preds

        return content_recs.sort_values('predicted_rating', ascending=False).head(top_k)
