"""
hybrid_recommender.py
----------------------
Cold-start-aware wrapper around the KNN collaborative filters.

Production KNN-based recommenders fail silently at the edges: a brand new
user has no rating row to compare against (user-user CF has nothing to
work with), and a brand new movie has no rating column (item-item CF has
nothing to work with). This module makes the fallback strategy EXPLICIT
rather than letting predict_rating() return None all the way to the API.

Fallback ladder (highest to lowest signal):
  1. Full collaborative filtering (ItemKNN and/or UserKNN) when the user has
     >= 1 rating and enough neighbors exist.
  2. Content-based similarity using movie feature vectors (genres, decade,
     MPAA, popularity) from feature_engineering.py — works even for movies
     with zero ratings, and for users we know a LITTLE about (1-2 ratings).
  3. Global popularity baseline (most-rated / highest-rated movies) — the
     universal fallback for a user we know NOTHING about (0 ratings) and
     for whom we have no stated preferences either.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

import config
from models.knn_recommender import ItemKNNRecommender, UserKNNRecommender, RatingMatrix


class HybridRecommender:
    def __init__(
        self,
        item_knn: ItemKNNRecommender,
        user_knn: UserKNNRecommender,
        rating_matrix: RatingMatrix,
        movie_features: np.ndarray,
        movies_df: pd.DataFrame,
        ratings_df: pd.DataFrame,
    ):
        self.item_knn = item_knn
        self.user_knn = user_knn
        self.rating_matrix = rating_matrix
        self.movie_features = movie_features
        self.movies_df = movies_df.reset_index(drop=True)
        self.movie_id_to_feat_idx = {mid: i for i, mid in enumerate(movies_df["movie_id"])}
        self._content_sim = cosine_similarity(movie_features)

        # Popularity baseline: mean rating weighted by rating count
        # (a simple Bayesian-ish shrinkage keeps low-count movies from
        # dominating with a single 5-star rating).
        agg = ratings_df.groupby("movie_id")["rating"].agg(["mean", "count"])
        global_mean = ratings_df["rating"].mean() if len(ratings_df) else 3.0
        m = 5  # shrinkage strength
        agg["bayesian_score"] = (agg["count"] * agg["mean"] + m * global_mean) / (agg["count"] + m)
        self.popularity_ranked = agg.sort_values("bayesian_score", ascending=False)
        self.global_mean_rating = global_mean

    # ---------------------------------------------------------------
    # /recommendations/{user_id}
    # ---------------------------------------------------------------
    def recommend_for_user(self, user_id: int, n: int = config.DEFAULT_N_RECOMMENDATIONS,
                            strategy: str = "item_knn") -> dict:
        """Returns a dict with the recommendation list AND which fallback
        tier was used, so the API/response is transparent about data
        quality (important for debugging and for UX messaging like
        'Because you're new, here's what's popular')."""
        rm = self.rating_matrix
        if user_id not in rm.user_index:
            return self._popularity_fallback(n, reason="unknown_user")

        u_idx = rm.user_index[user_id]
        n_ratings = int((rm.matrix[u_idx] != 0).sum())

        if n_ratings == 0:
            return self._popularity_fallback(n, reason="cold_start_zero_ratings")

        if n_ratings < 3:
            # A LITTLE signal: blend content-based similarity from the
            # movies they rated (however few) instead of relying on sparse
            # collaborative neighbors, which are unreliable at this count.
            return self._content_based_fallback(user_id, n, reason="cold_start_few_ratings")

        model = self.item_knn if strategy == "item_knn" else self.user_knn
        results = model.recommend_for_user(user_id, n=n)
        if len(results) < n:
            # Collaborative model ran out of signal (e.g. rated only
            # obscure movies) — top up with content-based candidates.
            backfill = self._content_based_fallback(
                user_id, n - len(results), reason="backfill",
                exclude_ids={mid for mid, _ in results},
            )["recommendations"]
            results = results + [(r["movie_id"], r["score"]) for r in backfill]

        return {
            "user_id": user_id,
            "strategy_used": strategy,
            "fallback_reason": None,
            "recommendations": [
                {"movie_id": int(mid), "score": round(float(score), 3)}
                for mid, score in results
            ],
        }

    def _content_based_fallback(self, user_id: int, n: int, reason: str,
                                 exclude_ids: set | None = None) -> dict:
        exclude_ids = exclude_ids or set()
        rm = self.rating_matrix
        u_idx = rm.user_index.get(user_id)
        rated_movie_idx = []
        if u_idx is not None:
            rated_movie_idx = list(np.where(rm.matrix[u_idx] != 0)[0])

        if not rated_movie_idx:
            return self._popularity_fallback(n, reason=reason)

        # Average the content-feature vectors of the user's rated movies,
        # weighted by their given rating, then find nearest movies to that
        # "taste centroid" in content-feature space.
        weights = rm.matrix[u_idx, rated_movie_idx]
        vectors = self.movie_features[rated_movie_idx]
        taste_centroid = np.average(vectors, axis=0, weights=weights).reshape(1, -1)

        sims = cosine_similarity(taste_centroid, self.movie_features)[0]
        rated_set = {rm.movie_ids[i] for i in rated_movie_idx} | exclude_ids
        ranked = sorted(
            ((self.movies_df.iloc[i]["movie_id"], sims[i]) for i in range(len(sims))
             if self.movies_df.iloc[i]["movie_id"] not in rated_set),
            key=lambda x: -x[1],
        )[:n]

        return {
            "user_id": user_id,
            "strategy_used": "content_based",
            "fallback_reason": reason,
            "recommendations": [
                {"movie_id": int(mid), "score": round(float(score), 3)} for mid, score in ranked
            ],
        }

    def _popularity_fallback(self, n: int, reason: str) -> dict:
        top = self.popularity_ranked.head(n)
        return {
            "user_id": None,
            "strategy_used": "popularity_baseline",
            "fallback_reason": reason,
            "recommendations": [
                {"movie_id": int(mid), "score": round(float(row["bayesian_score"]), 3)}
                for mid, row in top.iterrows()
            ],
        }

    # ---------------------------------------------------------------
    # /related_movies/{movie_id}
    # ---------------------------------------------------------------
    def similar_movies(self, movie_id: int, n: int = config.DEFAULT_N_RECOMMENDATIONS) -> dict:
        rm = self.rating_matrix
        has_ratings = (
            movie_id in rm.movie_index and
            (rm.matrix[:, rm.movie_index[movie_id]] != 0).sum() >= config.MIN_CO_RATINGS_FOR_PEARSON
        )

        if has_ratings:
            results = self.item_knn.similar_items(movie_id, n=n)
            return {
                "movie_id": movie_id,
                "strategy_used": "item_item_collaborative",
                "similar_movies": [
                    {"movie_id": int(mid), "similarity": round(float(sim), 3)} for mid, sim in results
                ],
            }

        # Cold-start movie (few/no ratings) -> content-based similarity
        if movie_id not in self.movie_id_to_feat_idx:
            return {"movie_id": movie_id, "strategy_used": "unknown_movie", "similar_movies": []}

        idx = self.movie_id_to_feat_idx[movie_id]
        sims = self._content_sim[idx].copy()
        sims[idx] = -np.inf
        top_idx = np.argsort(-sims)[:n]
        return {
            "movie_id": movie_id,
            "strategy_used": "content_based",
            "similar_movies": [
                {"movie_id": int(self.movies_df.iloc[i]["movie_id"]), "similarity": round(float(sims[i]), 3)}
                for i in top_idx
            ],
        }
