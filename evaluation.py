"""
evaluation.py
--------------
Task 4: Rigorous Model Evaluation (Model Selection).

Two complementary evaluation lenses, as requested:

1. Rating-prediction accuracy (regression-style):
     MAE, RMSE on held-out (user, movie, true_rating) triples.
     -> answers "how close are our predicted scores to the truth?"

2. Top-K ranking quality (recommendation-style):
     Precision@K, Recall@K on held-out RELEVANT items (rating >= threshold).
     -> answers "when we show the top K, how many are actually good?"

Train/test split strategy: PER-USER leave-N-out. We hold out
config.TEST_HOLDOUT_PER_USER ratings per user (only for users who have
enough ratings to make a fair split) and train on everything else. This is
the standard protocol for recommender evaluation — a random global split
would leak information and could leave some users with zero training
ratings, corrupting cold-start behavior we WANT to test explicitly.

The `compare_configurations` function is the model-selection entry point:
it fits every (algorithm x metric x k) combination on the same train split
and reports all metrics side-by-side, which is what a production ML
engineer would use to pick the deployed configuration.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import itertools
import json
import numpy as np
import pandas as pd

import config
from models.knn_recommender import (
    RatingMatrix, ItemKNNRecommender, UserKNNRecommender, build_rating_matrix,
)


def train_test_split_ratings(ratings_df: pd.DataFrame, n_holdout: int,
                              rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-N-out split, per user. Users with fewer than n_holdout + 2
    ratings are kept entirely in train (holding them out would leave no
    training signal and artificially tank cold-start-adjacent metrics)."""
    train_rows, test_rows = [], []
    for uid, group in ratings_df.groupby("user_id"):
        if len(group) < n_holdout + 2:
            train_rows.append(group)
            continue
        shuffled = group.sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000)))
        test_rows.append(shuffled.iloc[:n_holdout])
        train_rows.append(shuffled.iloc[n_holdout:])
    train_df = pd.concat(train_rows, ignore_index=True) if train_rows else ratings_df.iloc[0:0]
    test_df = pd.concat(test_rows, ignore_index=True) if test_rows else ratings_df.iloc[0:0]
    return train_df, test_df


# --------------------------------------------------------------------------
# Regression metrics: MAE / RMSE
# --------------------------------------------------------------------------

def evaluate_rating_prediction(model, rating_matrix: RatingMatrix, test_df: pd.DataFrame) -> dict:
    errors = []
    n_skipped = 0
    for row in test_df.itertuples():
        u_idx = rating_matrix.user_index.get(row.user_id)
        m_idx = rating_matrix.movie_index.get(row.movie_id)
        if u_idx is None or m_idx is None:
            n_skipped += 1
            continue
        pred = model.predict_rating(u_idx, m_idx)
        if pred is None:
            n_skipped += 1
            continue
        errors.append((pred, row.rating))

    if not errors:
        return {"mae": None, "rmse": None, "n_evaluated": 0, "n_skipped": n_skipped}

    preds = np.array([e[0] for e in errors])
    truths = np.array([e[1] for e in errors])
    mae = float(np.mean(np.abs(preds - truths)))
    rmse = float(np.sqrt(np.mean((preds - truths) ** 2)))
    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "n_evaluated": len(errors), "n_skipped": n_skipped}


# --------------------------------------------------------------------------
# Top-K ranking metrics: Precision@K / Recall@K
# --------------------------------------------------------------------------

def evaluate_top_k(model, rating_matrix: RatingMatrix, test_df: pd.DataFrame,
                    k: int = config.TOP_K_FOR_EVAL,
                    relevance_threshold: float = config.RELEVANCE_THRESHOLD) -> dict:
    """For each test user with >=1 relevant held-out item, generate a top-K
    recommendation list (from TRAIN-time model state) and measure overlap
    with the held-out relevant set."""
    relevant_by_user = (
        test_df[test_df["rating"] >= relevance_threshold]
        .groupby("user_id")["movie_id"]
        .apply(set)
        .to_dict()
    )

    precisions, recalls = [], []
    for user_id, relevant_movies in relevant_by_user.items():
        if user_id not in rating_matrix.user_index:
            continue
        recs = model.recommend_for_user(user_id, n=k)
        recommended_ids = {mid for mid, _ in recs}
        if not recommended_ids:
            continue
        hits = recommended_ids & relevant_movies
        precisions.append(len(hits) / len(recommended_ids))
        recalls.append(len(hits) / len(relevant_movies))

    return {
        "precision_at_k": round(float(np.mean(precisions)), 4) if precisions else None,
        "recall_at_k": round(float(np.mean(recalls)), 4) if recalls else None,
        "k": k,
        "n_users_evaluated": len(precisions),
    }


# --------------------------------------------------------------------------
# Model-selection sweep
# --------------------------------------------------------------------------

def compare_configurations(
    users_df: pd.DataFrame, movies_df: pd.DataFrame, ratings_df: pd.DataFrame,
    algorithms=("item_knn", "user_knn"),
    metrics=config.SIMILARITY_METRICS,
    k_values=(10, 20, 30),
    seed: int = config.RANDOM_SEED,
) -> list[dict]:
    """Fits every (algorithm x similarity metric x k) combination on an
    identical train/test split and reports MAE/RMSE + Precision@K/Recall@K
    for each — the side-by-side table a Senior ML Architect would use to
    select the production configuration."""
    rng = np.random.default_rng(seed)
    train_df, test_df = train_test_split_ratings(ratings_df, config.TEST_HOLDOUT_PER_USER, rng)
    rating_matrix = build_rating_matrix(train_df, users_df, movies_df)

    results = []
    for algo, metric, k in itertools.product(algorithms, metrics, k_values):
        if algo == "item_knn":
            model = ItemKNNRecommender(k=k, metric=metric).fit(rating_matrix)
        else:
            model = UserKNNRecommender(k=k, metric=metric).fit(rating_matrix)

        reg_metrics = evaluate_rating_prediction(model, rating_matrix, test_df)
        rank_metrics = evaluate_top_k(model, rating_matrix, test_df, k=config.TOP_K_FOR_EVAL)

        results.append({
            "algorithm": algo,
            "similarity_metric": metric,
            "k": k,
            **reg_metrics,
            **rank_metrics,
        })
        print(f"[eval] {algo:9s} | {metric:7s} | k={k:3d} | "
              f"MAE={reg_metrics['mae']} RMSE={reg_metrics['rmse']} | "
              f"P@{config.TOP_K_FOR_EVAL}={rank_metrics['precision_at_k']} "
              f"R@{config.TOP_K_FOR_EVAL}={rank_metrics['recall_at_k']}")

    return results


def select_best_configuration(results: list[dict], sort_by: str = "rmse") -> dict:
    """Picks the best config by the given metric (lower is better for
    mae/rmse, higher is better for precision/recall)."""
    valid = [r for r in results if r.get(sort_by) is not None]
    reverse = sort_by in ("precision_at_k", "recall_at_k")
    valid.sort(key=lambda r: r[sort_by], reverse=reverse)
    return valid[0] if valid else {}


if __name__ == "__main__":
    users_df = pd.read_csv(config.USERS_CSV)
    movies_df = pd.read_csv(config.MOVIES_CSV)
    ratings_df = pd.read_csv(config.RATINGS_CSV)

    results = compare_configurations(users_df, movies_df, ratings_df)
    best = select_best_configuration(results, sort_by="rmse")
    print("\nBest configuration by RMSE:", json.dumps(best, indent=2))

    with open(config.EVAL_REPORT_JSON, "w") as f:
        json.dump({"all_results": results, "best_by_rmse": best}, f, indent=2)
    print(f"\nSaved full evaluation report to {config.EVAL_REPORT_JSON}")
