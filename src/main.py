"""Minimal entrypoint and small demo runner for data preparation.

This module provides a `run_data_prep()` function that mirrors the
notebook steps: generate synthetic data, preprocess, and stratified split.
It allows running the same flow from the command line so tests and CI
can invoke it easily.
"""
from .config import BASE_DIR, DATA_DIR
from .logger import get_logger

from .data_loader import generate_synthetic_cic_data, preprocess, stratified_split

from pathlib import Path
import pandas as pd

logger = get_logger(__name__)


def run_data_prep(n_samples: int = 1000, categorical_cols=None, apply_smote: bool = False, df: pd.DataFrame | None = None):
    """Run the data preparation flow and return artifacts.

    Returns a dict with keys: df, out, splits
    """
    if categorical_cols is None:
        categorical_cols = ["protocol", "service"]

    # If a DataFrame is provided (or CSVs exist in DATA_DIR), use that. Otherwise generate synthetic data.
    if df is None:
        # look for CSVs in DATA_DIR
        data_path = Path(DATA_DIR)
        csv_files = sorted(data_path.glob("*.csv")) if data_path.exists() else []
        if csv_files:
            logger.info("Loading CSV files from %s: %s", DATA_DIR, [p.name for p in csv_files])
            parts = [pd.read_csv(p) for p in csv_files]
            df = pd.concat(parts, ignore_index=True)
        else:
            logger.info("Generating synthetic dataset (n=%d)", n_samples)
            df = generate_synthetic_cic_data(n_samples=n_samples)

    # Print a small sample to the log so CLI/CI output shows it
    logger.info("Sample rows:\n%s", df.head().to_string())

    out = preprocess(df, categorical_cols=categorical_cols, apply_smote=apply_smote)
    logger.info("Feature matrix shape: %s", out["X"].shape)
    logger.info("Label distribution:\n%s", out["y"].value_counts().to_string())

    splits = stratified_split(out["X"], out["y"], test_size=0.15, val_size=0.15)
    logger.info("Train/Val/Test shapes: %s %s %s", splits["X_train"].shape, splits["X_val"].shape, splits["X_test"].shape)

    return {"df": df, "out": out, "splits": splits}


def main():
    logger.info(f"BASE_DIR={BASE_DIR}")
    logger.info(f"DATA_DIR={DATA_DIR}")
    logger.info("Running data preparation demo")
    run_data_prep(n_samples=1000, apply_smote=False)


if __name__ == "__main__":
    main()
