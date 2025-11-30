from surprise import SVD, Reader, Dataset
from surprise.model_selection import train_test_split
from surprise import accuracy
import pickle
import pandas as pd
from collections import defaultdict
from src.utils import SVD_MODEL_PATH


class CollaborativeRecommender:
    def __init__(self, ratings=None):
        self.ratings = ratings
        self.model = None
        self.test_metrics = {}

    def train(self):
        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(
            self.ratings[['userId', 'movieId', 'rating']], reader)
        trainset, testset = train_test_split(
            data, test_size=0.2, random_state=42)

        print("Training SVD...")
        self.model = SVD(n_factors=50, n_epochs=20, random_state=42)
        self.model.fit(trainset)

        predictions = self.model.test(testset)
        self.test_metrics['RMSE'] = accuracy.rmse(predictions, verbose=False)
        self.test_metrics['MAE'] = accuracy.mae(predictions, verbose=False)

        # Calculate Precision/Recall
        self.test_metrics['Precision@10'], self.test_metrics['Recall@10'] = self.calculate_precision_recall(
            predictions)

        # Additional ranking metrics: NDCG@10, MAP@10, HitRate@10, Coverage
        ndcg10, map10, hit10, coverage = self.calculate_additional_metrics(
            predictions)
        self.test_metrics['NDCG@10'] = ndcg10
        self.test_metrics['MAP@10'] = map10
        self.test_metrics['HitRate@10'] = hit10
        self.test_metrics['Coverage'] = coverage

    def calculate_precision_recall(self, predictions, k=10, threshold=3.5):
        user_est_true = defaultdict(list)
        for uid, _, true_r, est, _ in predictions:
            user_est_true[uid].append((est, true_r))

        precisions, recalls = {}, {}
        for uid, user_ratings in user_est_true.items():
            user_ratings.sort(key=lambda x: x[0], reverse=True)
            n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
            n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
            n_rel_and_rec_k = sum(((true_r >= threshold) and (
                est >= threshold)) for (est, true_r) in user_ratings[:k])
            precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
            recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

        return sum(prec for prec in precisions.values()) / len(precisions), sum(rec for rec in recalls.values()) / len(recalls)

    # --- Additional ranking metrics ---
    def _ndcg_at_k(self, rels, k):
        """Compute NDCG@k for a list of binary relevances (rels)."""
        import math
        dcg = 0.0
        for i, rel in enumerate(rels[:k]):
            denom = math.log2(i + 2)
            dcg += (2 ** rel - 1) / denom
        ideal_rels = sorted(rels, reverse=True)[:k]
        idcg = 0.0
        for i, rel in enumerate(ideal_rels):
            denom = math.log2(i + 2)
            idcg += (2 ** rel - 1) / denom
        return 0.0 if idcg == 0 else dcg / idcg

    def _average_precision_at_k(self, rels, k):
        """Compute AP@k for a list of binary relevances."""
        num_rel = 0
        score = 0.0
        for i, rel in enumerate(rels[:k]):
            if rel:
                num_rel += 1
                score += num_rel / (i + 1)
        return score / min(sum(rels), k) if sum(rels) > 0 else 0.0

    def calculate_additional_metrics(self, predictions, k=10, threshold=3.5):
        """Calculate NDCG@k, MAP@k, HitRate@k and Coverage."""
        from collections import defaultdict

        user_est_true = defaultdict(list)
        for uid, iid, true_r, est, _ in predictions:
            user_est_true[uid].append((iid, est, true_r))

        ndcgs = []
        maps = []
        hits = []
        all_recommended = set()

        for uid, vals in user_est_true.items():
            vals.sort(key=lambda x: x[1], reverse=True)
            rels = [1 if (true_r >= threshold)
                    else 0 for (_, _, true_r) in vals]
            topk_ids = [iid for (iid, _, _) in vals[:k]]
            all_recommended.update(topk_ids)

            ndcgs.append(self._ndcg_at_k(rels, k))
            maps.append(self._average_precision_at_k(rels, k))
            hits.append(1.0 if any(r == 1 for r in rels[:k]) else 0.0)

        try:
            total_items = len(self.ratings['movieId'].unique())
            coverage = len(all_recommended) / \
                total_items if total_items > 0 else 0.0
        except Exception:
            coverage = 0.0

        ndcg_mean = sum(ndcgs) / len(ndcgs) if ndcgs else 0.0
        map_mean = sum(maps) / len(maps) if maps else 0.0
        hitrate = sum(hits) / len(hits) if hits else 0.0

        return ndcg_mean, map_mean, hitrate, coverage

    def get_top_n_for_user(self, user_id, movies_df, n=10):
        """
        Gets the top-N movie recommendations for a single user.
        """
        # Get list of all movie IDs
        all_movie_ids = movies_df['movieId'].unique()

        # If we don't have a ratings DataFrame, or it doesn't contain
        # userId, we can't compute per-user recommendations. Return
        # a safe fallback (popular movies) instead.
        if self.ratings is None or self.ratings.empty or 'userId' not in self.ratings.columns:
            try:
                top_by_count = movies_df.sort_values(
                    'popularity', ascending=False).head(n)
                return top_by_count
            except Exception:
                return movies_df.head(n)

        # Get list of movies the user has already rated
        rated_movie_ids = self.ratings[self.ratings['userId']
                                       == user_id]['movieId'].unique()

        # If the provided user_id is unknown to the ratings set, return
        # a simple fallback: the top-n movies by popularity in movies_df
        if user_id not in self.ratings['userId'].unique():
            # attempt to return the top-n by rating count if available
            try:
                top_by_count = movies_df.sort_values(
                    'popularity', ascending=False).head(n)
                return top_by_count
            except Exception:
                return movies_df.head(n)

        # Get movies the user has not yet rated
        unrated_movie_ids = [
            mid for mid in all_movie_ids if mid not in rated_movie_ids]

        # Predict ratings for unrated movies (safe: catch prediction exceptions)
        predictions = []
        for mid in unrated_movie_ids:
            try:
                pred = self.model.predict(user_id, mid)
            except Exception:
                # If the model can't predict for this mid, skip it
                continue
            predictions.append(pred)

        # Sort predictions by estimated rating
        predictions.sort(key=lambda x: x.est, reverse=True)

        # Get top N movie IDs
        top_n_preds = predictions[:n]
        top_n_movie_ids = [pred.iid for pred in top_n_preds]

        # Return the movie details
        # Preserve recommendation order: create ordered DataFrame
        if len(top_n_movie_ids) == 0:
            return pd.DataFrame()

        ordered = movies_df.set_index(
            'movieId').loc[top_n_movie_ids].reset_index()
        # if movieId lookup fails (e.g., missing ids), fall back to isin
        if ordered.empty:
            return movies_df[movies_df['movieId'].isin(top_n_movie_ids)]
        return ordered

    def save(self):
        with open(SVD_MODEL_PATH, 'wb') as f:
            pickle.dump({'model': self.model, 'metrics': self.test_metrics}, f)

    def load(self):
        """
        Load saved SVD model. If ratings_df is supplied in the constructor
        it will be used; otherwise ratings remain whatever was previously set.
        """
        with open(SVD_MODEL_PATH, 'rb') as f:
            data = pickle.load(f)
            self.model = data.get('model', None)
            self.test_metrics = data.get('metrics', {})

    def predict(self, user_id, movie_id):
        # Safe predict: return neutral score if model missing or prediction fails
        if self.model is None:
            return 3.0
        try:
            return float(self.model.predict(user_id, movie_id).est)
        except Exception:
            return 3.0
