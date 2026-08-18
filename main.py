"""
api/main.py
-----------
Task 5: Real-Time Inference API.

A production-style FastAPI service exposing the trained HybridRecommender.

Endpoints:
  GET /health                          -> liveness check
  GET /recommendations/{user_id}       -> personalized recommendations
  GET /related_movies/{movie_id}       -> content/collaborative "more like this"
  GET /movies/{movie_id}               -> metadata lookup (helper for API consumers)

Design notes:
  - The model artifact is loaded ONCE at process startup (not per-request),
    which is the standard pattern for low-latency inference.
  - Query parameters (k, strategy) let callers tune behavior without a
    redeploy.
  - Every response includes `strategy_used` / `fallback_reason` so
    downstream consumers (and on-call engineers) can see exactly which
    cold-start tier served the request.
  - Errors return proper HTTP status codes (404 for unknown ids) rather
    than empty 200s, per REST best practice.

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from contextlib import asynccontextmanager

# إضافة المسار عشان يقدر يقرا ملف config
sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config

# --------------------------------------------------------------------------
# Model loading (startup)
# --------------------------------------------------------------------------

_state: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not config.MODEL_PICKLE.exists():
        raise RuntimeError(
            f"No trained model found at {config.MODEL_PICKLE}. "
            f"Run `python train_pipeline.py` first."
        )
    with open(config.MODEL_PICKLE, "rb") as f:
        artifact = pickle.load(f)
    _state["hybrid"] = artifact["hybrid_recommender"]
    _state["movies_df"] = artifact["movies_df"].set_index("movie_id")
    _state["users_df"] = artifact["users_df"].set_index("user_id")
    _state["config_used"] = artifact["config_used"]
    print(f"[api] Model loaded. Trained config: {artifact['config_used']}")
    yield
    _state.clear()


# --------------------------------------------------------------------------
# App Initialization (دمجنا التعريف هنا مرة واحدة بس)
# --------------------------------------------------------------------------

app = FastAPI(
    title="Movie Recommendation Engine API",
    description="KNN-based collaborative filtering with content-based cold-start fallback.",
    version="1.0.0",
    lifespan=lifespan,
)

# السماح للواجهة بالاتصال بالسيرفر (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Response schemas
# --------------------------------------------------------------------------

class RecommendationItem(BaseModel):
    movie_id: int
    title: str | None = None
    score: float


class RecommendationResponse(BaseModel):
    user_id: int | None
    strategy_used: str
    fallback_reason: str | None
    recommendations: list[RecommendationItem]


class SimilarMovieItem(BaseModel):
    movie_id: int
    title: str | None = None
    similarity: float


class RelatedMoviesResponse(BaseModel):
    movie_id: int
    title: str | None = None
    strategy_used: str
    similar_movies: list[SimilarMovieItem]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _movie_title(movie_id: int) -> str | None:
    movies_df = _state.get("movies_df")
    if movies_df is not None and movie_id in movies_df.index:
        return str(movies_df.loc[movie_id, "title"])
    return None


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "model_config": _state.get("config_used")}


@app.get("/recommendations/{user_id}", response_model=RecommendationResponse)
def get_recommendations(
    user_id: int,
    k: int = Query(config.DEFAULT_N_RECOMMENDATIONS, ge=1, le=50,
                   description="Number of recommendations to return"),
    strategy: str = Query("item_knn", pattern="^(item_knn|user_knn)$",
                          description="Collaborative filtering strategy to prefer"),
):
    hybrid = _state["hybrid"]
    result = hybrid.recommend_for_user(user_id, n=k, strategy=strategy)

    for item in result["recommendations"]:
        item["title"] = _movie_title(item["movie_id"])

    return result


@app.get("/related_movies/{movie_id}", response_model=RelatedMoviesResponse)
def get_related_movies(
    movie_id: int,
    k: int = Query(config.DEFAULT_N_RECOMMENDATIONS, ge=1, le=50,
                   description="Number of similar movies to return"),
):
    movies_df = _state["movies_df"]
    if movie_id not in movies_df.index:
        raise HTTPException(status_code=404, detail=f"movie_id {movie_id} not found")

    hybrid = _state["hybrid"]
    result = hybrid.similar_movies(movie_id, n=k)
    result["title"] = _movie_title(movie_id)
    for item in result["similar_movies"]:
        item["title"] = _movie_title(item["movie_id"])

    return result


@app.get("/movies/{movie_id}")
def get_movie(movie_id: int):
    movies_df = _state["movies_df"]
    if movie_id not in movies_df.index:
        raise HTTPException(status_code=404, detail=f"movie_id {movie_id} not found")
    row = movies_df.loc[movie_id]
    return {
        "movie_id": movie_id,
        "title": row["title"],
        "genres": row["genres"].split("|"),
        "decade": int(row["decade"]),
        "mpaa_rating": row["mpaa_rating"],
        "popularity": float(row["popularity"]),
    }