"""Data acquisition and preprocessing utilities for AI-IDS (MVP).

This module provides:
- Synthetic data generator for quick testing.
- Cleaning utilities (drop duplicates, fill missing values).
- Encoding (one-hot via pandas.get_dummies) and scaling (Z-score).
- Optional SMOTE balancing.
- Stratified train/val/test splitting.

Designed for easy adaptation to real datasets (CIC-IDS2017, NSL-KDD, UNSW-NB15).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple, List, Optional, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib

from .config import DATA_DIR, MODELS_DIR, RANDOM_SEED
from .logger import get_logger

logger = get_logger(__name__)


def generate_synthetic_cic_data(n_samples: int = 2000, imbalance: float = 0.95, random_state: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate a small synthetic dataset resembling network flow features.

    Columns:
    - SrcBytes, DstBytes, FlowDuration, TotPkts: numeric
    - protocol: categorical (TCP/UDP/ICMP)
    - service: categorical (HTTP/FTP/SSH/SMTP)
    - label: 'Benign' or 'Attack'

    imbalance: fraction of benign samples (0..1)
    """
    rng = np.random.default_rng(random_state)

    n_benign = int(n_samples * imbalance)
    n_attack = n_samples - n_benign

    def flows(n, attack=False):
        if not attack:
            src_bytes = rng.normal(loc=500, scale=200, size=n).clip(0)
            dst_bytes = rng.normal(loc=400, scale=180, size=n).clip(0)
            duration = rng.exponential(scale=1000, size=n)
            pkts = rng.poisson(lam=10, size=n)
            proto = rng.choice(["TCP", "UDP", "ICMP"], size=n, p=[0.7, 0.25, 0.05])
            service = rng.choice(["HTTP", "SSH", "FTP", "SMTP"], size=n, p=[0.6, 0.15, 0.15, 0.1])
        else:
            src_bytes = rng.normal(loc=5000, scale=4000, size=n).clip(0)
            dst_bytes = rng.normal(loc=3000, scale=2500, size=n).clip(0)
            duration = rng.exponential(scale=3000, size=n)
            pkts = rng.poisson(lam=100, size=n)
            proto = rng.choice(["TCP", "UDP", "ICMP"], size=n, p=[0.5, 0.45, 0.05])
            service = rng.choice(["HTTP", "SSH", "FTP", "SMTP"], size=n, p=[0.4, 0.2, 0.3, 0.1])

        return pd.DataFrame({
            "SrcBytes": src_bytes,
            "DstBytes": dst_bytes,
            "FlowDuration": duration,
            "TotPkts": pkts,
            "protocol": proto,
            "service": service,
        })

    df_b = flows(n_benign, attack=False)
    df_b["label"] = "Benign"

    df_a = flows(n_attack, attack=True)
    df_a["label"] = "Attack"

    df = pd.concat([df_b, df_a], ignore_index=True).sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Introduce some missing values randomly for robustness
    for col in ["SrcBytes", "DstBytes", "service"]:
        mask = rng.random(df.shape[0]) < 0.01
        df.loc[mask, col] = np.nan

    return df


def clean_data(df: pd.DataFrame, numeric_fill: Optional[str] = "median") -> pd.DataFrame:
    """Clean raw DataFrame: drop duplicates, fill missing numeric and categorical values."""
    logger.info("Cleaning data: dropping duplicates and filling missing values")
    df = df.copy()
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    after = len(df)
    logger.debug("Dropped %d duplicated rows", before - after)

    # Fill numeric
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in numeric_cols:
        if df[c].isna().any():
            if numeric_fill == "median":
                val = df[c].median()
            elif numeric_fill == "mean":
                val = df[c].mean()
            else:
                val = 0
            df[c].fillna(val, inplace=True)

    # Fill categorical
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for c in cat_cols:
        if df[c].isna().any():
            mode = df[c].mode()
            if not mode.empty:
                df[c].fillna(mode.iloc[0], inplace=True)
            else:
                df[c].fillna("unknown", inplace=True)

    return df


def encode_and_scale(df: pd.DataFrame, categorical_cols: Optional[List[str]] = None, scaler: Optional[StandardScaler] = None) -> Tuple[pd.DataFrame, StandardScaler, List[str]]:
    """One-hot encode categorical columns using pandas.get_dummies and apply StandardScaler to numeric features.

    Returns: (X_df, scaler, feature_columns)
    """
    df = df.copy()
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    else:
        # Filter requested categorical columns to those actually present in df
        present = [c for c in categorical_cols if c in df.columns]
        missing = [c for c in categorical_cols if c not in df.columns]
        if missing:
            logger.warning("Requested categorical columns %s not found in DataFrame — ignoring: %s", categorical_cols, missing)
        categorical_cols = present

    # exclude label if present
    if "label" in categorical_cols:
        categorical_cols = [c for c in categorical_cols if c != "label"]

    # If no categorical columns remain, let pandas infer by passing None
    cols_for_dummies = categorical_cols if (categorical_cols and len(categorical_cols) > 0) else None

    logger.info("Encoding categorical columns: %s", cols_for_dummies)
    df = pd.get_dummies(df, columns=cols_for_dummies, drop_first=False)

    feature_cols = [c for c in df.columns if c != "label"]

    X = df[feature_cols]
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols, index=df.index)
    return X_scaled_df, scaler, feature_cols


def preprocess(df: pd.DataFrame, label_col: str = "label", categorical_cols: Optional[List[str]] = None, apply_smote: bool = False, random_state: int = RANDOM_SEED) -> Dict[str, object]:
    """Full preprocessing pipeline for tabular IDS data.

    Returns dictionary with keys: X, y, scaler, feature_columns
    """
    df_clean = clean_data(df)

    if label_col not in df_clean.columns:
        # Attempt a safe fallback: infer label from a Filename column if present.
        # Many exported sample CSVs include a `Filename` field like
        # 'Spyware-...-<id>.raw' — we can take the prefix before the first
        # separator as a coarse label. This keeps the pipeline usable when
        # CSVs don't include an explicit 'label' column.
        if "Filename" in df_clean.columns:
            logger.warning("Label column '%s' not found — inferring labels from 'Filename' column", label_col)
            inferred = df_clean["Filename"].astype(str).str.split("[-_.]").str[0]
            # Write inferred values into the desired label_col name
            df_clean[label_col] = inferred
        else:
            raise ValueError(f"Label column '{label_col}' not found in DataFrame")

    y = df_clean[label_col].apply(lambda x: 1 if str(x).lower() != "benign" and str(x).lower() != "normal" else 0)

    X_df, scaler, feature_cols = encode_and_scale(df_clean.drop(columns=[label_col]), categorical_cols=categorical_cols)

    if apply_smote:
        logger.info("Applying SMOTE to balance classes")
        sm = SMOTE(random_state=random_state)
        X_res, y_res = sm.fit_resample(X_df, y)
        X_df = pd.DataFrame(X_res, columns=X_df.columns)
        y = pd.Series(y_res)
        logger.info("After SMOTE class distribution: %s", y.value_counts().to_dict())
    else:
        logger.info("Class distribution: %s", y.value_counts().to_dict())

    return {"X": X_df, "y": y, "scaler": scaler, "feature_columns": feature_cols}


def stratified_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.15, val_size: float = 0.15, random_state: int = RANDOM_SEED) -> Dict[str, object]:
    """Split X and y into train/val/test with stratification.

    Note: val_size is proportion of the original dataset (not of train).
    """
    # First split out test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

    # Now split temp into train and val. Adjust val_size to proportion of temp
    val_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_adjusted, stratify=y_temp, random_state=random_state)

    logger.info("Split sizes — train: %d, val: %d, test: %d", len(X_train), len(X_val), len(X_test))

    return {"X_train": X_train, "X_val": X_val, "X_test": X_test, "y_train": y_train, "y_val": y_val, "y_test": y_test}


def save_preprocessor(scaler: StandardScaler, feature_columns: List[str], path: Optional[Path] = None) -> Path:
    """Save scaler and feature columns for later inference."""
    if path is None:
        path = Path(MODELS_DIR) / "preprocessor.joblib"
    payload = {"scaler": scaler, "feature_columns": feature_columns}
    joblib.dump(payload, path)
    logger.info("Saved preprocessor to %s", path)
    return path


if __name__ == "__main__":
    # Quick demo when running this module directly
    df = generate_synthetic_cic_data(1000)
    out = preprocess(df, categorical_cols=["protocol", "service"], apply_smote=False)
    splits = stratified_split(out["X"], out["y"], test_size=0.15, val_size=0.15)
    print("Train shape:", splits["X_train"].shape)
    print("Val shape:", splits["X_val"].shape)
    print("Test shape:", splits["X_test"].shape)
    save_preprocessor(out["scaler"], out["feature_columns"])