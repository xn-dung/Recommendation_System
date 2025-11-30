import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dropout
from tensorflow.keras.models import Model, load_model
import pandas as pd
import numpy as np
import pickle
from src.utils import DL_MODEL_PATH, MAPPINGS_PATH


class DeepLearningRecommender:
    def __init__(self, num_users=None, num_movies=None, mappings=None):
        self.mappings = mappings
        if num_users and num_movies:
            self.model = self._build_model(num_users, num_movies)
        else:
            self.model = None

    def _build_model(self, n_users, n_movies):
        # Neural Collaborative Filtering (NCF) Architecture
        user_input = Input(shape=(1,), name='user_input')
        user_emb = Embedding(n_users, 50, name='user_embedding')(user_input)
        user_vec = Flatten()(user_emb)

        movie_input = Input(shape=(1,), name='movie_input')
        movie_emb = Embedding(
            n_movies, 50, name='movie_embedding')(movie_input)
        movie_vec = Flatten()(movie_emb)
        concat = Concatenate()([user_vec, movie_vec])

        dense1 = Dense(128, activation='relu')(concat)
        dropout1 = Dropout(0.2)(dense1)
        dense2 = Dense(64, activation='relu')(dropout1)
        out = Dense(1)(dense2)

        model = Model(inputs=[user_input, movie_input], outputs=out)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def train(self, ratings, epochs=5):
        print("Training Deep Learning Model (NCF)...")
        self.model.fit(
            [ratings['user_encoded'], ratings['movie_encoded']],
            ratings['rating'],
            batch_size=1024,
            epochs=epochs,
            verbose=1,
            validation_split=0.1
        )

    def save(self):
        self.model.save(DL_MODEL_PATH)
        with open(MAPPINGS_PATH, 'wb') as f:
            pickle.dump(self.mappings, f)

    def load(self):
        # Load model without compiling to avoid issues deserializing
        # loss/metric functions (some saved models reference names
        # that change across TF/Keras versions). We'll re-compile
        # explicitly after loading.
        try:
            self.model = load_model(DL_MODEL_PATH, compile=False)
            # Re-compile with the same settings used at training time
            self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        except Exception:
            # Fallback: try loading using custom_objects mapping if
            # the model contains names that can't be resolved.
            from tensorflow.keras import losses, metrics
            custom = {
                'mse': losses.mean_squared_error,
                'mae': metrics.mean_absolute_error,
            }
            self.model = load_model(DL_MODEL_PATH, custom_objects=custom)
        with open(MAPPINGS_PATH, 'rb') as f:
            self.mappings = pickle.load(f)

    def predict(self, user_id, movie_id):
        if self.mappings is None:
            return 3.0

        u = self.mappings['u2u'].get(user_id)
        m = self.mappings['m2m'].get(movie_id)

        if u is None or m is None:
            return 3.0

        return float(self.model.predict([np.array([u]), np.array([m])], verbose=0)[0][0])
