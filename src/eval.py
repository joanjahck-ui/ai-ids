"""Evaluation utilities for AI-IDS.

Compute standard classification metrics and print/save results.
"""
from __future__ import annotations

from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def compute_metrics(y_true, y_pred, y_prob=None) -> Dict[str, Any]:
    metrics = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)

    # ROC AUC requires probability scores or decision function
    try:
        if y_prob is not None:
            # assume positive class is at column 1
            if y_prob.ndim == 2:
                probs = y_prob[:, 1]
            else:
                probs = y_prob
            metrics["roc_auc"] = roc_auc_score(y_true, probs)
        else:
            metrics["roc_auc"] = None
    except Exception:
        metrics["roc_auc"] = None

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics["confusion_matrix"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    # False positive rate = FP / (FP + TN)
    metrics["fpr"] = float(fp) / float(fp + tn) if (fp + tn) > 0 else 0.0

    return metrics


def pretty_print_metrics(metrics: Dict[str, Any]):
    print("=== Evaluation Metrics ===")
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "fpr"]:
        print(f"{k}: {metrics.get(k)}")
    print("Confusion Matrix:", metrics.get("confusion_matrix"))
