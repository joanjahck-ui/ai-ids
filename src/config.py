from pathlib import Path
import os

# Base project paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
REPORTS_DIR = BASE_DIR / "reports"

# Reproducibility
RANDOM_SEED = 42

# Ensure directories exist
for p in (DATA_DIR, MODELS_DIR, LOGS_DIR, REPORTS_DIR):
    os.makedirs(p, exist_ok=True)
