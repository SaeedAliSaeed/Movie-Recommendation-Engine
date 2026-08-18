"""
train_pipeline.py
------------------
End-to-end orchestration script. Run this once to:
  1. Generate (or load) mock data.
  2. Build movie content features (TF-IDF + one-hot + PCA).
  3. Build the user-item rating matrix.
  4. Run the evaluation sweep across algorithms/metrics/k (model selection).
  5. Fit the FINAL production model using the best configuration found.
  6. Wrap it in the cold-start-aware HybridRecommender.
  7. Serialize everything the API needs to a single pickle artifact.

Usage:
    python train_pipeline.py                # full run (regenerates data)
    python train_pipeline.py --skip-data     # reuse existing CSVs in artifacts/
    python train_pipeline.py --skip-eval     # skip the sweep, use config defaults
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
import re

import pandas as pd

import config
from data.feature_engineering import build_movie_feature_matrix
from models.knn_recommender import build_rating_matrix, ItemKNNRecommender, UserKNNRecommender
from models.hybrid_recommender import HybridRecommender
from models.evaluation import compare_configurations, select_best_configuration

def main(skip_eval: bool = False):
    t0 = time.time()

    movies_df = pd.read_csv("artifacts/movies.csv")
    ratings_df = pd.read_csv("artifacts/ratings.csv")

    movies_df.rename(columns={"movieId": "movie_id"}, inplace=True)
    ratings_df.rename(columns={"userId": "user_id", "movieId": "movie_id"}, inplace=True)

    def extract_decade(title):
        match = re.search(r'\((\d{4})\)', title)
        if match:
            year = int(match.group(1))
            return (year // 10) * 10
        return 2000

    movies_df['decade'] = movies_df['title'].apply(extract_decade)
    movies_df['mpaa_rating'] = 'NR'

    popularity_counts = ratings_df.groupby('movie_id').size().reset_index(name='popularity')
    movies_df = pd.merge(movies_df, popularity_counts, on='movie_id', how='left')
    movies_df['popularity'] = movies_df['popularity'].fillna(1.0)

    users_df = pd.DataFrame({"user_id": ratings_df["user_id"].unique()})

    movie_features, feature_engineer = build_movie_feature_matrix(movies_df)

    if skip_eval:
        best_algo, best_metric, best_k = "item_knn", "cosine", config.DEFAULT_K_NEIGHBORS
    else:
        results = compare_configurations(users_df, movies_df, ratings_df)
        best = select_best_configuration(results, sort_by="rmse")
        best_algo, best_metric, best_k = best["algorithm"], best["similarity_metric"], best["k"]
        with open(config.EVAL_REPORT_JSON, "w") as f:
            json.dump({"all_results": results, "best_by_rmse": best}, f, indent=2)

    rating_matrix = build_rating_matrix(ratings_df, users_df, movies_df)
    item_knn = ItemKNNRecommender(k=best_k, metric=best_metric).fit(rating_matrix)
    user_knn = UserKNNRecommender(k=best_k, metric=best_metric).fit(rating_matrix)

    hybrid = HybridRecommender(
        item_knn=item_knn,
        user_knn=user_knn,
        rating_matrix=rating_matrix,
        movie_features=movie_features,
        movies_df=movies_df,
        ratings_df=ratings_df,
    )

    artifact = {
        "hybrid_recommender": hybrid,
        "movies_df": movies_df,
        "users_df": users_df,
        "feature_engineer": feature_engineer,
        "config_used": {"algorithm": best_algo, "metric": best_metric, "k": best_k},
    }
    with open(config.MODEL_PICKLE, "wb") as f:
        pickle.dump(artifact, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()
    main(skip_eval=args.skip_eval)