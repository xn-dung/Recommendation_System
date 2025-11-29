from surprise import SVD, Reader, Dataset
from surprise.model_selection import train_test_split
from surprise import accuracy
import pickle
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

    def get_top_n_for_user(self, user_id, movies_df, n=10):
        """
        Gets the top-N movie recommendations for a single user.
        """
        # Get list of all movie IDs
        all_movie_ids = movies_df['movieId'].unique()

        # Get list of movies the user has already rated
        rated_movie_ids = self.ratings[self.ratings['userId']
                                       == user_id]['movieId'].unique()

        # Get movies the user has not yet rated
        unrated_movie_ids = [
            mid for mid in all_movie_ids if mid not in rated_movie_ids]

        # Predict ratings for unrated movies
        predictions = [self.model.predict(user_id, mid)
                       for mid in unrated_movie_ids]

        # Sort predictions by estimated rating
        predictions.sort(key=lambda x: x.est, reverse=True)

        # Get top N movie IDs
        top_n_preds = predictions[:n]
        top_n_movie_ids = [pred.iid for pred in top_n_preds]

        # Return the movie details
        return movies_df[movies_df['movieId'].isin(top_n_movie_ids)]

    def save(self):
        with open(SVD_MODEL_PATH, 'wb') as f:
            pickle.dump({'model': self.model, 'metrics': self.test_metrics}, f)

    def load(self):
        with open(SVD_MODEL_PATH, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.test_metrics = data.get('metrics', {})

    def predict(self, user_id, movie_id):
        return self.model.predict(user_id, movie_id).est
