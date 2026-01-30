from __future__ import annotations

import re
from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)


def _safe_key(name: str) -> str:
    """
    Make a stable key for JSON/CSV columns.
    """
    return re.sub(r"[^0-9a-zA-Z_]+", "_", str(name)).strip("_")


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    class_names: List[str],
) -> Dict[str, float]:
    """
    Returns a flat dict of metrics for tables.

    Overall:
      - acc
      - macro_f1, weighted_f1
      - macro_precision, macro_recall
      - weighted_precision, weighted_recall

    Per-class (for each class name C):
      - precision_C, recall_C, f1_C, support_C
      - class_acc_C  (same as recall_C; common "per-class accuracy")
    """
    assert len(class_names) == num_classes, "class_names length must match num_classes"

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    labels = list(range(num_classes))

    metrics: Dict[str, float] = {}

    # -------- overall metrics --------
    metrics["acc"] = float(accuracy_score(y_true, y_pred))

    metrics["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))
    metrics["weighted_f1"] = float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0))

    metrics["macro_precision"] = float(precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))
    metrics["macro_recall"] = float(recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))

    metrics["weighted_precision"] = float(precision_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0))
    metrics["weighted_recall"] = float(recall_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0))

    # -------- per-class metrics --------
    p, r, f1, sup = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )

    for i, cname in enumerate(class_names):
        k = _safe_key(cname)
        metrics[f"precision_{k}"] = float(p[i])
        metrics[f"recall_{k}"] = float(r[i])
        metrics[f"f1_{k}"] = float(f1[i])
        metrics[f"support_{k}"] = float(sup[i])  # keep as float for consistent aggregation
        metrics[f"class_acc_{k}"] = float(r[i])

    return metrics


def compute_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    labels = list(range(num_classes))
    return confusion_matrix(y_true, y_pred, labels=labels)
