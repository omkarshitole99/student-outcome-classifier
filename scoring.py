"""Metric helpers shared by train_models.py and the Streamlit page.

The six metric names come directly from the assignment PDF. Precision, recall
and F1 are macro averages, so the smaller 'Enrolled' group counts as much as
the larger 'Graduate' group. AUC is the macro one-vs-rest AUC taken from
predict_proba. Accuracy and MCC are the ordinary multi-class versions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRICS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def predict_with(pipeline, features, labels):
    """Predicted labels, plus probabilities with columns in `labels` order."""
    predicted = pipeline.predict(features)
    proba = pipeline.predict_proba(features)
    slot = {name: i for i, name in enumerate(pipeline.classes_)}
    return predicted, proba[:, [slot[name] for name in labels]]


def macro_ovr_auc(actual, proba, labels) -> float:
    """Average of the one-vs-rest AUC of each outcome.

    An outcome that is absent from `actual` (or the only one present) has no
    AUC, so it is skipped rather than crashing on a small uploaded file.
    """
    actual = np.asarray(actual)
    per_outcome = []
    for column, name in enumerate(labels):
        is_outcome = actual == name
        if is_outcome.all() or not is_outcome.any():
            continue
        per_outcome.append(roc_auc_score(is_outcome, proba[:, column]))
    return float(np.mean(per_outcome)) if per_outcome else float("nan")


def score_model(actual, predicted, proba, labels) -> dict:
    return {
        "Accuracy": float(accuracy_score(actual, predicted)),
        "AUC": macro_ovr_auc(actual, proba, labels),
        "Precision": float(precision_score(actual, predicted, average="macro", zero_division=0)),
        "Recall": float(recall_score(actual, predicted, average="macro", zero_division=0)),
        "F1": float(f1_score(actual, predicted, average="macro", zero_division=0)),
        "MCC": float(matthews_corrcoef(actual, predicted)),
    }


def confusion_counts(actual, predicted, labels) -> pd.DataFrame:
    grid = confusion_matrix(actual, predicted, labels=list(labels))
    return pd.DataFrame(grid, index=list(labels), columns=list(labels))


def per_class_report(actual, predicted, labels) -> pd.DataFrame:
    report = classification_report(
        actual, predicted, labels=list(labels), output_dict=True, zero_division=0
    )
    report.pop("accuracy", None)  # shown separately as a headline number
    return pd.DataFrame(report).T.round(4)
