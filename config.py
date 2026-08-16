"""
config.py
---------
Central configuration for the Movie Recommendation Engine.
Keeping all tunable parameters in one place makes the pipeline easy to
reproduce and to hand off to another engineer (a core production-readiness
requirement).
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

USERS_CSV = ARTIFACT_DIR / "users.csv"
MOVIES_CSV = ARTIFACT_DIR / "movies.csv"
RATINGS_CSV = ARTIFACT_DIR / "ratings.csv"
MODEL_PICKLE = ARTIFACT_DIR / "recommender.pkl"
EVAL_REPORT_JSON = ARTIFACT_DIR / "evaluation_report.json"

# --------------------------------------------------------------------------
# Mock data simulation
# --------------------------------------------------------------------------
N_USERS = 600
N_MOVIES = 350
RANDOM_SEED = 42

# Genre vocabulary — deliberately includes the genre mixes called out in the
# brief (action-comedy, crime, spy thrillers, sci-fi) plus enough breadth
# that TF-IDF / similarity signals are meaningful rather than trivial.
GENRES = [
    "Action", "Comedy", "Action-Comedy", "Crime", "Spy-Thriller",
    "Sci-Fi", "Drama", "Romance", "Horror", "Animation", "Thriller",
    "Mystery", "Fantasy", "Documentary",
]

DECADES = [1980, 1990, 2000, 2010, 2020]
MPAA_RATINGS = ["G", "PG", "PG-13", "R", "NC-17"]

# Target sparsity of the user-item rating matrix. Real-world catalogs
# (Netflix, MovieLens-25M) sit well under 5% density; we mimic that here.
RATING_MATRIX_DENSITY = 0.06

# Cold-start simulation knobs
COLD_START_NEW_USER_FRACTION = 0.08   # users with 0-2 ratings ("new users")
COLD_START_NEW_MOVIE_FRACTION = 0.05  # movies with 0 ratings ("new movies")
MIN_RATINGS_ESTABLISHED_USER = 8

# --------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------
TFIDF_MAX_FEATURES = 60          # cap TF-IDF vocabulary from genre "soup"
PCA_VARIANCE_RETAINED = 0.95     # PCA keeps components explaining 95% var
PCA_MIN_DIM_TO_TRIGGER = 15      # only run PCA if raw feature dim exceeds this

# --------------------------------------------------------------------------
# KNN model
# --------------------------------------------------------------------------
DEFAULT_K_NEIGHBORS = 20
SIMILARITY_METRICS = ["cosine", "pearson"]   # supported similarity metrics
MIN_CO_RATINGS_FOR_PEARSON = 3     # min overlap required for a stable pearson corr

# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
TEST_HOLDOUT_PER_USER = 3         # leave-N-out ratings held out per user
RELEVANCE_THRESHOLD = 4.0         # rating >= this counts as "relevant" for Precision/Recall@K
TOP_K_FOR_EVAL = 10

# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------
API_HOST = "0.0.0.0"
API_PORT = 8000
DEFAULT_N_RECOMMENDATIONS = 10
