r"""Training script for baseline ML models (Random Forest, XGBoost, SVM).

This script uses the synthetic data generator and preprocessing pipeline.
It performs a small GridSearchCV for each model (lightweight for demo), fits the best
model on training data, evaluates on test data, and saves the trained models.

Run from project root:
    python ./src/train_ml.py

"""

import sys
from pathlib import Path
from pprint import pprint


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Check runtime requirements before importing heavy ML/data libraries so we can
# provide a helpful error message instead of crashing with ModuleNotFoundError.
from src.requirement_checks import check_requirements, format_install_instructions
missing = check_requirements(["numpy", "pandas", "sklearn", "imbalanced_learn", "xgboost"])
if missing:
    print(format_install_instructions(missing))
    raise SystemExit(1)

from src.data_loader import generate_synthetic_cic_data, preprocess, stratified_split, save_preprocessor
from src.models import RandomForestModel, XGBoostModel, SVMModel
from src.eval import compute_metrics, pretty_print_metrics

from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.utils import compute_class_weight
from src.config import RANDOM_SEED
from src.logger import get_logger

logger = get_logger(__name__)


def run_demo():
    # Generate synthetic data and preprocess
    df = generate_synthetic_cic_data(n_samples=2000)
    out = preprocess(df, categorical_cols=["protocol", "service"], apply_smote=False)
    splits = stratified_split(out["X"], out["y"], test_size=0.15, val_size=0.15)

    X_train = splits["X_train"]
    y_train = splits["y_train"]
    X_val = splits["X_val"]
    y_val = splits["y_val"]
    X_test = splits["X_test"]
    y_test = splits["y_test"]

    # Save preprocessor for inference
    save_preprocessor(out["scaler"], out["feature_columns"])

    # Define models and small param grids for demo
    models = []

    rf = RandomForestModel()
    rf_grid = {"n_estimators": [50, 100], "max_depth": [None, 10]}
    models.append(("RandomForest", rf, rf_grid))

    xg = XGBoostModel(use_label_encoder=False)
    xgb_grid = {"n_estimators": [50, 100], "max_depth": [3, 6]}
    models.append(("XGBoost", xg, xgb_grid))

    svm = SVMModel()
    svm_grid = {"C": [0.1, 1.0], "kernel": ["rbf"]}
    models.append(("SVM", svm, svm_grid))

    results = {}

    for name, wrapper, grid in models:
        print(f"\n=== Training {name} ===")
        # Wrap model.model into GridSearchCV
        clf = GridSearchCV(wrapper.model, grid, cv=5, scoring="f1", n_jobs=1)
        clf.fit(X_train, y_train)
        print("Best params:")
        pprint(clf.best_params_)

        best_model = clf.best_estimator_
        wrapper.model = best_model
        # Evaluate on test
        y_pred = wrapper.predict(X_test)
        try:
            y_prob = wrapper.predict_proba(X_test)
        except Exception:
            y_prob = None

        metrics = compute_metrics(y_test, y_pred, y_prob)
        pretty_print_metrics(metrics)

        model_path = Path(wrapper.save())
        print(f"Saved model to: {model_path}")
        results[name] = {"metrics": metrics, "path": str(model_path)}

    # Summary
    print("\n=== Summary Results ===")
    pprint(results)


if __name__ == "__main__":
    run_demo()
