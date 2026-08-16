# Movie Recommendation Engine — KNN-Based Collaborative Filtering

A production-style, end-to-end personalized recommendation system built around
**K-Nearest-Neighbors collaborative filtering**, with content-based and
popularity fallbacks for cold-start users/movies, a rigorous evaluation
framework, and a real-time FastAPI inference layer.

## Architecture

```
movie_recommender/
├── config.py                     # all tunable parameters, single source of truth
├── data/
│   ├── data_generator.py         # Task 1: mock users/movies/ratings + cold-start simulation
│   └── feature_engineering.py    # Task 2: TF-IDF + one-hot + PCA on movie metadata
├── models/
│   ├── knn_recommender.py        # Task 3: ItemKNN + UserKNN (cosine & pearson), parameterized K
│   ├── hybrid_recommender.py     # cold-start fallback ladder (collab -> content -> popularity)
│   └── evaluation.py             # Task 4: MAE/RMSE + Precision@K/Recall@K, config sweep
├── api/
│   └── main.py                   # Task 5: FastAPI service (/recommendations, /related_movies)
├── train_pipeline.py              # orchestrates the whole pipeline end-to-end
├── artifacts/                     # generated: CSVs, evaluation_report.json, recommender.pkl
└── requirements.txt
```

### Why this design

**Data (Task 1).** Users get a latent Dirichlet-sampled genre-affinity
vector; movies get 1–3 genres drawn from a pool weighted toward
action-comedy, crime, spy-thriller, and sci-fi (as requested). Ratings are
sampled from a genre-affinity-driven signal + noise, at a controlled ~3–6%
matrix density, matching real-world sparsity (Netflix/MovieLens sit well
under 5%). Cold start is simulated explicitly: a fraction of users get 0–2
ratings ("new users") and a fraction of movies get zero ratings ("new
movies") — the system must handle both.

**Features (Task 2).** Genres go through TF-IDF (not plain one-hot) so
common genres are down-weighted and discriminative ones stand out. Decade
and MPAA rating are one-hot encoded. PCA is applied *conditionally* —
only if combined dimensionality exceeds a threshold — and keeps 95% of
variance, avoiding both unnecessary compression and the curse of
dimensionality in KNN distance computations.

**Collaborative filtering (Task 3).** Both `ItemKNNRecommender` (movies
similar to what a user rated) and `UserKNNRecommender` (users with similar
taste) are implemented from scratch on top of `sklearn`-style similarity
functions, supporting **cosine** and **Pearson** similarity (Pearson =
cosine on mean-centered rows — centering removes each user's/item's rating
bias before comparing preference *shape*). `k` is a constructor argument
everywhere, so it's swept during evaluation rather than hardcoded.

**Evaluation (Task 4).** Uses a per-user leave-N-out train/test split
(the standard recommender-evaluation protocol — a random global split can
strip a user of all training signal and corrupts cold-start testing).
Reports both **regression accuracy** (MAE, RMSE on held-out ratings) and
**ranking quality** (Precision@K, Recall@K against held-out "relevant"
items, rating ≥ 4). `compare_configurations()` sweeps
`{item_knn, user_knn} × {cosine, pearson} × {k=10,20,30}` and
`select_best_configuration()` picks the deployed config.

**Cold start.** `HybridRecommender` wraps the KNN models in an explicit
fallback ladder:
1. Full collaborative filtering, when the user has enough ratings.
2. Content-based similarity (movie feature vectors) — works for movies with
   zero ratings and users with only 1–2 ratings.
3. Bayesian-shrunk popularity baseline — the universal fallback for
   completely unknown users.
Every API response includes `strategy_used` and `fallback_reason` so
callers (and on-call engineers) know exactly what tier served the request.

**API (Task 5).** FastAPI loads the trained artifact once at startup
(not per-request). Endpoints:
- `GET /recommendations/{user_id}?k=10&strategy=item_knn|user_knn`
- `GET /related_movies/{movie_id}?k=10`
- `GET /movies/{movie_id}` (metadata helper)
- `GET /health`

## How to run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the full pipeline (generates mock data, engineers features,
#    runs the evaluation sweep, fits the production model, saves artifacts)
python train_pipeline.py

#    Subsequent runs can reuse existing data / skip the sweep:
python train_pipeline.py --skip-data
python train_pipeline.py --skip-data --skip-eval

# 3. Launch the API
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# 4. Test it
curl "http://localhost:8000/health"
curl "http://localhost:8000/recommendations/1?k=5&strategy=item_knn"
curl "http://localhost:8000/related_movies/1?k=5"
```

Interactive API docs (Swagger UI) are auto-served at `http://localhost:8000/docs`.

## Running individual components / tests

```bash
python data/data_generator.py          # regenerate mock data only
python data/feature_engineering.py     # inspect the movie feature matrix
python models/evaluation.py            # run just the evaluation sweep, writes evaluation_report.json
```

## Extending to a real dataset

Swap `data/data_generator.py`'s output for real `users.csv` / `movies.csv`
/ `ratings.csv` with the same column schema (`user_id`, `movie_id`,
`rating`, `genres`, etc.) and the rest of the pipeline (feature engineering,
KNN models, evaluation, API) requires no changes — this is why the modules
are decoupled from the data-generation step via CSV artifacts.

## Known scaling considerations for production

- The current similarity matrices are dense (`O(n^2)`), fine for
  thousands of users/items in a demo. At real scale, replace
  `sklearn.metrics.pairwise.cosine_similarity` with
  `sklearn.neighbors.NearestNeighbors` (already imported) using
  `algorithm="brute"` + sparse input, or move to an ANN index
  (FAISS / ScaNN) for sub-linear neighbor lookup.
- Retraining is currently a full batch job (`train_pipeline.py`). For
  production, schedule it (e.g. nightly) and consider incremental
  similarity updates for high-velocity catalogs.
- The rating matrix is held in memory as a dense NumPy array for
  simplicity; a real deployment should use `scipy.sparse` throughout.
