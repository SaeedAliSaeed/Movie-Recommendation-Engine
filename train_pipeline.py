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

import pandas as pd

import config
from data.data_generator import build_mock_dataset
from data.feature_engineering import build_movie_feature_matrix
from models.knn_recommender import build_rating_matrix, ItemKNNRecommender, UserKNNRecommender
from models.hybrid_recommender import HybridRecommender
from models.evaluation import compare_configurations, select_best_configuration


def main(skip_data: bool = False, skip_eval: bool = False):
    t0 = time.time()

    # ---------------------------------------------------------------
    # 1. Data
    # ---------------------------------------------------------------
    if skip_data and config.USERS_CSV.exists():
        print("[pipeline] Loading existing mock data from artifacts/ ...")
        users_df = pd.read_csv(config.USERS_CSV)
        movies_df = pd.read_csv(config.MOVIES_CSV)
        ratings_df = pd.read_csv(config.RATINGS_CSV)
    else:
        print("[pipeline] Generating mock data ...")
        users_df, movies_df, ratings_df = build_mock_dataset(save=True)

    # ---------------------------------------------------------------
    # 2. Feature engineering (movie content features)
    # ---------------------------------------------------------------
    print("[pipeline] Building movie content features ...")
    movie_features, feature_engineer = build_movie_feature_matrix(movies_df)

    # ---------------------------------------------------------------
    # 3. Model selection sweep (Task 4)
    # ---------------------------------------------------------------
    if skip_eval:
        best_algo, best_metric, best_k = "item_knn", "cosine", config.DEFAULT_K_NEIGHBORS
        print(f"[pipeline] Skipping eval sweep, using defaults: "
              f"{best_algo}/{best_metric}/k={best_k}")
    else:
        print("[pipeline] Running evaluation sweep for model selection ...")
        results = compare_configurations(users_df, movies_df, ratings_df)
        best = select_best_configuration(results, sort_by="rmse")
        best_algo, best_metric, best_k = best["algorithm"], best["similarity_metric"], best["k"]
        print(f"[pipeline] Best config by RMSE: {best_algo} / {best_metric} / k={best_k}")
        with open(config.EVAL_REPORT_JSON, "w") as f:
            json.dump({"all_results": results, "best_by_rmse": best}, f, indent=2)

    # ---------------------------------------------------------------
    # 4. Fit FINAL production models on the FULL rating data
    #    (evaluation sweep used a held-out split; production model uses
    #    all available signal).
    # ---------------------------------------------------------------
    print("[pipeline] Fitting final production models on full data ...")
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

    # ---------------------------------------------------------------
    # 5. Serialize for the API
    # ---------------------------------------------------------------
    artifact = {
        "hybrid_recommender": hybrid,
        "movies_df": movies_df,
        "users_df": users_df,
        "feature_engineer": feature_engineer,
        "config_used": {"algorithm": best_algo, "metric": best_metric, "k": best_k},
    }
    with open(config.MODEL_PICKLE, "wb") as f:
        pickle.dump(artifact, f)

    elapsed = time.time() - t0
    print(f"[pipeline] Saved production artifact -> {config.MODEL_PICKLE}")
    print(f"[pipeline] Done in {elapsed:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-data", action="store_true", help="Reuse existing CSVs")
    parser.add_argument("--skip-eval", action="store_true", help="Skip the evaluation sweep")
    args = parser.parse_args()
    main(skip_data=args.skip_data, skip_eval=args.skip_eval)
