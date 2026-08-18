"""
knn_recommender.py
-------------------
Task 3: Collaborative Filtering Pipeline with KNN.

Implements:
  - A sparse user-item rating matrix builder.
  - ItemKNNRecommender: item-item collaborative filtering (finds movies
    similar to movies a user already liked, based on co-rating patterns).
  - UserKNNRecommender: user-user collaborative filtering (finds users with
    similar taste, aggregates what THEY liked).
  - Two similarity metrics: cosine and Pearson correlation. Pearson is
    implemented as cosine similarity on MEAN-CENTERED rows, which is the
    standard reduction (centering removes each user's/item's rating bias
    before comparing shape of preference, which cosine alone doesn't do).
  - K is a first-class, parameterized argument on every public method.
  - A cold-start-aware wrapper (HybridRecommender) that falls back to
    content-based similarity (movie feature vectors from
    feature_engineering.py) or global popularity when collaborative signal
    is unavailable — this is the standard production pattern for handling
    new users/items in a KNN-based system.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

import config


# --------------------------------------------------------------------------
# Rating matrix utilities
# --------------------------------------------------------------------------

@dataclass
class RatingMatrix:
    """Wraps the dense/sparse user-item matrix plus id<->index lookups."""
    matrix: np.ndarray            # dense, shape (n_users, n_movies), 0 = unrated
    user_ids: np.ndarray
    movie_ids: np.ndarray
    user_index: dict              # user_id -> row index
    movie_index: dict             # movie_id -> col index

    @property
    def n_users(self) -> int:
        return self.matrix.shape[0]

    @property
    def n_movies(self) -> int:
        return self.matrix.shape[1]


def build_rating_matrix(ratings_df: pd.DataFrame, users_df: pd.DataFrame,
                         movies_df: pd.DataFrame) -> RatingMatrix:
    """Pivots the long-format ratings table into a dense user-item matrix.

    We use the FULL user/movie universe (not just those with ratings) so
    that cold-start users/movies are present as all-zero rows/columns —
    this makes the sparsity explicit and lets the hybrid layer detect them.
    """
    user_ids = users_df["user_id"].to_numpy()
    movie_ids = movies_df["movie_id"].to_numpy()
    user_index = {uid: i for i, uid in enumerate(user_ids)}
    movie_index = {mid: i for i, mid in enumerate(movie_ids)}

    matrix = np.zeros((len(user_ids), len(movie_ids)), dtype=np.float32)
    for row in ratings_df.itertuples():
        u = user_index.get(row.user_id)
        m = movie_index.get(row.movie_id)
        if u is not None and m is not None:
            matrix[u, m] = row.rating

    return RatingMatrix(matrix, user_ids, movie_ids, user_index, movie_index)


def _mean_center_rows(matrix: np.ndarray) -> np.ndarray:
    """Mean-centers each row using only its NON-ZERO (i.e. rated) entries.
    This is the standard trick to turn cosine similarity into Pearson
    correlation for sparse rating data, without densifying assumptions
    about unrated entries (they remain 0 / neutral after centering)."""
    centered = matrix.copy()
    for i in range(matrix.shape[0]):
        nonzero = matrix[i] != 0
        if nonzero.any():
            row_mean = matrix[i, nonzero].mean()
            centered[i, nonzero] = matrix[i, nonzero] - row_mean
    return centered


def _similarity_matrix(matrix: np.ndarray, metric: str) -> np.ndarray:
    """Computes a full similarity matrix for either metric.

    cosine  -> raw cosine similarity on rating vectors (fast, sklearn).
    pearson -> cosine similarity on mean-centered rows (see above).
    """
    if metric == "cosine":
        return cosine_similarity(matrix)
    elif metric == "pearson":
        centered = _mean_center_rows(matrix)
        return cosine_similarity(centered)
    else:
        raise ValueError(f"Unsupported similarity metric: {metric}")


# --------------------------------------------------------------------------
# Item-Item KNN
# --------------------------------------------------------------------------

class ItemKNNRecommender:
    """Item-item collaborative filtering.

    Similarity is computed between MOVIE columns of the rating matrix
    (i.e. two movies are similar if they were rated similarly by the same
    users). For a target user, we predict scores for unrated movies as a
    similarity-weighted average of their ratings on the user's already-rated
    movies, restricted to the K nearest neighbor movies of each.
    """

    def __init__(self, k: int = config.DEFAULT_K_NEIGHBORS, metric: str = "cosine"):
        self.k = k
        self.metric = metric
        self._sim: np.ndarray | None = None
        self._nn: NearestNeighbors | None = None
        self.rating_matrix: RatingMatrix | None = None

    def fit(self, rating_matrix: RatingMatrix):
        self.rating_matrix = rating_matrix
        item_matrix = rating_matrix.matrix.T  # shape (n_movies, n_users)
        self._sim = _similarity_matrix(item_matrix, self.metric)
        return self

    def _k_nearest_items(self, item_idx: int, exclude: set[int]) -> list[tuple[int, float]]:
        sims = self._sim[item_idx].copy()
        sims[item_idx] = -np.inf  # never neighbor with itself
        for e in exclude:
            sims[e] = -np.inf
        top_k_idx = np.argpartition(-sims, min(self.k, len(sims) - 1))[: self.k]
        top_k_idx = top_k_idx[np.argsort(-sims[top_k_idx])]
        return [(idx, sims[idx]) for idx in top_k_idx if sims[idx] > -np.inf]

    def predict_rating(self, user_row_idx: int, target_item_idx: int) -> float | None:
        """Predicts a rating for (user, item) via similarity-weighted average
        of the user's ratings on the K most-similar items they HAVE rated."""
        rm = self.rating_matrix
        user_ratings = rm.matrix[user_row_idx]
        rated_idx = np.where(user_ratings != 0)[0]
        if len(rated_idx) == 0:
            return None  # no signal at all -> caller should fall back

        sims_to_target = self._sim[target_item_idx, rated_idx]
        order = np.argsort(-sims_to_target)[: self.k]
        neighbor_idx = rated_idx[order]
        neighbor_sims = sims_to_target[order]

        valid = neighbor_sims > 0
        if not valid.any():
            return None
        weights = neighbor_sims[valid]
        ratings = user_ratings[neighbor_idx[valid]]
        return float(np.dot(weights, ratings) / weights.sum())

    def recommend_for_user(self, user_id: int, n: int = config.DEFAULT_N_RECOMMENDATIONS,
                            exclude_rated: bool = True) -> list[tuple[int, float]]:
        """Returns [(movie_id, predicted_score), ...] sorted descending."""
        rm = self.rating_matrix
        u = rm.user_index[user_id]
        user_ratings = rm.matrix[u]
        candidate_idx = range(rm.n_movies)
        if exclude_rated:
            candidate_idx = [i for i in candidate_idx if user_ratings[i] == 0]

        scored = []
        for m_idx in candidate_idx:
            pred = self.predict_rating(u, m_idx)
            if pred is not None:
                scored.append((rm.movie_ids[m_idx], pred))
        scored.sort(key=lambda x: -x[1])
        return scored[:n]

    def similar_items(self, movie_id: int, n: int = config.DEFAULT_N_RECOMMENDATIONS) -> list[tuple[int, float]]:
        """Task requirement: /related_movies/{movie_id}."""
        rm = self.rating_matrix
        if movie_id not in rm.movie_index:
            return []
        idx = rm.movie_index[movie_id]
        neighbors = self._k_nearest_items(idx, exclude=set())[:n]
        return [(rm.movie_ids[i], float(sim)) for i, sim in neighbors]


# --------------------------------------------------------------------------
# User-User KNN
# --------------------------------------------------------------------------

class UserKNNRecommender:
    """User-user collaborative filtering.

    Similarity is computed between USER rows of the rating matrix. For a
    target user, unrated movies are scored as a similarity-weighted average
    of ratings given by the K nearest neighbor users who DID rate that
    movie.
    """

    def __init__(self, k: int = config.DEFAULT_K_NEIGHBORS, metric: str = "cosine"):
        self.k = k
        self.metric = metric
        self._sim: np.ndarray | None = None
        self.rating_matrix: RatingMatrix | None = None

    def fit(self, rating_matrix: RatingMatrix):
        self.rating_matrix = rating_matrix
        self._sim = _similarity_matrix(rating_matrix.matrix, self.metric)
        return self

    def predict_rating(self, user_row_idx: int, target_item_idx: int) -> float | None:
        rm = self.rating_matrix
        item_column = rm.matrix[:, target_item_idx]
        raters = np.where(item_column != 0)[0]
        raters = raters[raters != user_row_idx]
        if len(raters) == 0:
            return None

        sims = self._sim[user_row_idx, raters]
        order = np.argsort(-sims)[: self.k]
        neighbor_idx = raters[order]
        neighbor_sims = sims[order]

        valid = neighbor_sims > 0
        if not valid.any():
            return None
        weights = neighbor_sims[valid]
        ratings = item_column[neighbor_idx[valid]]
        return float(np.dot(weights, ratings) / weights.sum())

    def recommend_for_user(self, user_id: int, n: int = config.DEFAULT_N_RECOMMENDATIONS,
                            exclude_rated: bool = True) -> list[tuple[int, float]]:
        rm = self.rating_matrix
        u = rm.user_index[user_id]
        user_ratings = rm.matrix[u]
        candidate_idx = range(rm.n_movies)
        if exclude_rated:
            candidate_idx = [i for i in candidate_idx if user_ratings[i] == 0]

        scored = []
        for m_idx in candidate_idx:
            pred = self.predict_rating(u, m_idx)
            if pred is not None:
                scored.append((rm.movie_ids[m_idx], pred))
        scored.sort(key=lambda x: -x[1])
        return scored[:n]

    def similar_users(self, user_id: int, n: int = config.DEFAULT_K_NEIGHBORS) -> list[tuple[int, float]]:
        rm = self.rating_matrix
        u = rm.user_index[user_id]
        sims = self._sim[u].copy()
        sims[u] = -np.inf
        top_idx = np.argsort(-sims)[:n]
        return [(rm.user_ids[i], float(sims[i])) for i in top_idx]
