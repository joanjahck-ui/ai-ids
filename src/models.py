"""ML model wrappers for AI-IDS.

Provides simple classes with train/predict/save/load for:
- RandomForest
- XGBoost
- SVM (with probability calibration via probability parameter)

Each class exposes a consistent interface so training scripts can use them interchangeably.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import joblib
from typing import Optional, Any, Dict

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb

from .config import MODELS_DIR, RANDOM_SEED


@dataclass
class BaseModel:
    name: str
    model: Any

    def fit(self, X, y, **kwargs):
        return self.model.fit(X, y, **kwargs)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        # For models without predict_proba, try decision_function
        if hasattr(self.model, "decision_function"):
            import numpy as np

            scores = self.model.decision_function(X)
            # convert to pseudo-proba via min-max
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
            return np.vstack([1 - scores, scores]).T
        raise NotImplementedError("Model has no method to produce probabilities")

    def save(self, path: Optional[Path] = None) -> Path:
        if path is None:
            path = Path(MODELS_DIR) / f"{self.name}.joblib"
        joblib.dump(self.model, path)
        return path

    def load(self, path: Path):
        self.model = joblib.load(path)
        return self


class RandomForestModel(BaseModel):
    def __init__(self, **kwargs):
        params = {"n_estimators": 100, "random_state": RANDOM_SEED}
        params.update(kwargs)
        model = RandomForestClassifier(**params)
        super().__init__(name="random_forest", model=model)


class XGBoostModel(BaseModel):
    def __init__(self, **kwargs):
        params = {"use_label_encoder": False, "eval_metric": "logloss", "random_state": RANDOM_SEED}
        params.update(kwargs)
        model = xgb.XGBClassifier(**params)
        super().__init__(name="xgboost", model=model)


class SVMModel(BaseModel):
    def __init__(self, **kwargs):
        params = {"probability": True, "random_state": RANDOM_SEED}
        params.update(kwargs)
        model = SVC(**params)
        super().__init__(name="svm", model=model)
