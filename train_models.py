"""Tune and save the five classifiers for the student outcome data.

Run from this folder:   python train_models.py
(model_training.ipynb runs the same steps one cell at a time.)

Reads data/uci_students_dropout.csv, sets 20% of the rows aside as
test_data.csv, tunes every model with cross-validation on the other 80%,
then scores each tuned model once on the rows it never saw. Models go to
model/*.joblib, numbers to model/training_summary.json and
model/holdout_metrics.csv. The Streamlit page only loads these files.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from prep import (
    RAW_FEATURES,
    TARGET,
    add_progress_features,
    build_preprocessor,
    input_columns,
    tidy_columns,
)
from scoring import METRICS, per_class_report, predict_with, score_model

HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "data" / "uci_students_dropout.csv"
MODEL_DIR = HERE / "model"

SEED = 5278
TEST_SHARE = 0.20
FOLDS = 5

# Decided from check_extra_features() below: see feature_check in the summary.
USE_EXTRA = True

# The assignment PDF fixes the five model families. The settings below are not
# copied from a public notebook; they are small grids chosen after checking this
# dataset's size, class imbalance and mixed feature types.
#
# Every grid key starts with "clf__" because the classifier is the second step
# of a Pipeline (the first step is the shared preprocessing).
MODELS = {
    "Logistic Regression": {
        "file": "logistic_regression.joblib",
        "estimator": LogisticRegression(max_iter=4000),
        "grid": {
            "clf__C": [0.03, 0.1, 0.3, 1.0, 3.0],
            "clf__class_weight": [None, "balanced"],
        },
    },
    "Decision Tree": {
        "file": "decision_tree.joblib",
        "estimator": DecisionTreeClassifier(random_state=SEED),
        "grid": {
            "clf__max_depth": [3, 4, 5, 6, 8, 10],
            "clf__min_samples_leaf": [5, 10, 20, 40],
            "clf__criterion": ["gini", "entropy"],
            "clf__class_weight": [None, "balanced"],
        },
    },
    "KNN": {
        "file": "knn.joblib",
        "estimator": KNeighborsClassifier(),
        "grid": {
            "clf__n_neighbors": [5, 9, 15, 25, 35],
            "clf__weights": ["uniform", "distance"],
            "clf__p": [1, 2],
        },
    },
    "Naive Bayes": {
        "file": "naive_bayes.joblib",
        "estimator": GaussianNB(),
        # First pass picked 0.1, the top of the grid, so 0.3 to 3 were added.
        "grid": {"clf__var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3, 1e-2, 1e-1, 3e-1, 1.0, 3.0]},
    },
    "Support Vector Machine": {
        "file": "svm.joblib",
        # SVC has no usable predict_proba any more, so after the grid search the
        # best SVC is wrapped in a calibrator. That gives probabilities for AUC.
        "calibrate": True,
        "estimator": SVC(kernel="rbf"),
        # First pass picked C=10 and gamma=0.003, both at the edge of the grid,
        # so the grid was shifted to cover larger C and smaller gamma.
        "grid": {
            "clf__C": [1.0, 3.0, 10.0, 30.0, 100.0],
            "clf__gamma": ["scale", 0.0003, 0.001, 0.003, 0.01],
            "clf__class_weight": [None, "balanced"],
        },
    },
}


def plain(value):
    """numpy numbers -> python numbers so json.dumps accepts them."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def load_table() -> pd.DataFrame:
    table = tidy_columns(pd.read_csv(DATA_FILE, sep=";"))
    expected = set(RAW_FEATURES) | {TARGET}
    if set(table.columns) != expected:
        extra = sorted(set(table.columns) - expected)
        absent = sorted(expected - set(table.columns))
        raise RuntimeError(f"Column mismatch. Unexpected: {extra}. Missing: {absent}.")
    return table


def split_table(table: pd.DataFrame):
    """80/20 split, stratified so both parts keep the same outcome mix."""
    return train_test_split(
        table, test_size=TEST_SHARE, stratify=table[TARGET], random_state=SEED
    )


def model_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    """The raw feature columns plus the three added progress columns."""
    return add_progress_features(frame[RAW_FEATURES])


def check_extra_features(x_train, y_train, folds) -> dict:
    """Does adding the pass-rate / grade-change columns help? Two quick baselines."""
    baselines = {
        "Logistic Regression": LogisticRegression(C=0.3, max_iter=4000, class_weight="balanced"),
        "SVM (rbf, defaults)": SVC(kernel="rbf", class_weight="balanced"),
    }
    out = {}
    for label, estimator in baselines.items():
        row = {}
        for use_extra in (False, True):
            pipe = Pipeline(
                [("prep", build_preprocessor(use_extra)), ("clf", clone(estimator))]
            )
            scores = cross_val_score(
                pipe,
                x_train[input_columns(use_extra)],
                y_train,
                cv=folds,
                scoring="f1_macro",
                n_jobs=-1,
            )
            row["with_extra_columns" if use_extra else "raw_columns_only"] = round(
                float(scores.mean()), 4
            )
        out[label] = row
    return out


def fit_model(spec, x_train, y_train, folds):
    """Tune one classifier family with GridSearchCV on the training rows only."""
    pipe = Pipeline(
        [("prep", build_preprocessor(USE_EXTRA)), ("clf", clone(spec["estimator"]))]
    )
    search = GridSearchCV(pipe, spec["grid"], scoring="f1_macro", cv=folds, n_jobs=-1)
    search.fit(x_train, y_train)
    chosen = search.best_estimator_
    if spec.get("calibrate"):
        calibrated = CalibratedClassifierCV(
            clone(chosen.named_steps["clf"]), method="sigmoid", cv=FOLDS, ensemble=False
        )
        chosen = Pipeline([("prep", build_preprocessor(USE_EXTRA)), ("clf", calibrated)])
        chosen.fit(x_train, y_train)
    return chosen, search


def top_features(name, model, count=10):
    """Names and weights for the two models whose inner workings are easy to read."""
    names = model.named_steps["prep"].get_feature_names_out()
    clf = model.named_steps["clf"]
    if name == "Decision Tree":
        weight = clf.feature_importances_
    elif name == "Logistic Regression":
        # signed: positive pushes towards Dropout, negative away from it
        weight = clf.coef_[list(clf.classes_).index("Dropout")]
    else:
        return None
    order = np.argsort(-np.abs(weight))[:count]
    return [{"feature": str(names[i]), "weight": round(float(weight[i]), 4)} for i in order]


def tune_and_score(name, x_train, y_train, x_test, y_test, labels, folds) -> dict:
    """Grid-search one model on the training rows, then score it once on the test rows."""
    spec = MODELS[name]
    model, search = fit_model(spec, x_train, y_train, folds)
    predicted, proba = predict_with(model, x_test[input_columns(USE_EXTRA)], labels)
    metrics = score_model(y_test, predicted, proba, labels)
    report = per_class_report(y_test, predicted, labels)
    record = {
        "file": spec["file"],
        "best_params": {
            key.replace("clf__", ""): plain(val) for key, val in search.best_params_.items()
        },
        "grid_size": int(len(search.cv_results_["params"])),
        "cv_macro_f1": round(float(search.best_score_), 4),
        "holdout": {key: round(value, 4) for key, value in metrics.items()},
        "holdout_f1_by_class": {label: float(report.loc[label, "f1-score"]) for label in labels},
        "top_features": top_features(name, model),
    }
    return {"model": model, "record": record, "metrics": metrics, "predicted": predicted}


def write_outputs(table, train, test, labels, feature_check, fitted) -> pd.DataFrame:
    """Save test_data.csv, the five models, the metrics table and the summary the app reads."""
    MODEL_DIR.mkdir(exist_ok=True)
    test.to_csv(HERE / "test_data.csv", index=False)
    for item in fitted.values():
        joblib.dump(item["model"], MODEL_DIR / item["record"]["file"])

    metrics_table = pd.DataFrame(
        [{"ML Model Name": name, **item["metrics"]} for name, item in fitted.items()]
    ).set_index("ML Model Name")[METRICS]
    metrics_table.round(4).to_csv(MODEL_DIR / "holdout_metrics.csv")

    y_train, y_test = train[TARGET], test[TARGET]
    summary = {
        "dataset": {
            "source": "UCI Machine Learning Repository, id 697 (Predict Students' Dropout and Academic Success)",
            "rows": int(len(table)),
            "raw_features": len(RAW_FEATURES),
            "class_counts": {k: int(v) for k, v in table[TARGET].value_counts().items()},
        },
        "split": {
            "seed": SEED,
            "test_share": TEST_SHARE,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_class_counts": {k: int(v) for k, v in y_train.value_counts().items()},
            "test_class_counts": {k: int(v) for k, v in y_test.value_counts().items()},
        },
        "labels": labels,
        "feature_columns": input_columns(USE_EXTRA),
        "tuning": {"folds": FOLDS, "scoring": "f1_macro"},
        "feature_check": feature_check,
        "models": {name: item["record"] for name, item in fitted.items()},
    }
    (MODEL_DIR / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return metrics_table


def main() -> None:
    table = load_table()
    labels = sorted(table[TARGET].unique())
    print(f"{len(table)} rows, {len(RAW_FEATURES)} raw features, classes: {labels}")

    train, test = split_table(table)
    x_train, x_test = model_inputs(train), model_inputs(test)
    y_train, y_test = train[TARGET], test[TARGET]
    folds = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

    print("\nChecking the added columns (5-fold macro-F1 on the training rows)")
    feature_check = check_extra_features(x_train, y_train, folds)
    for label, row in feature_check.items():
        print(f"  {label}: {row}")

    fitted = {}
    for name in MODELS:
        started = time.time()
        fitted[name] = tune_and_score(name, x_train, y_train, x_test, y_test, labels, folds)
        record, metrics = fitted[name]["record"], fitted[name]["metrics"]
        print(
            f"\n{name}  ({time.time() - started:.0f}s, {record['grid_size']} settings)"
            f"\n  best: {record['best_params']}"
            f"\n  cv macro-F1 {record['cv_macro_f1']:.4f}   hold-out F1 {metrics['F1']:.4f}"
            f"   acc {metrics['Accuracy']:.4f}   AUC {metrics['AUC']:.4f}"
        )

    metrics_table = write_outputs(table, train, test, labels, feature_check, fitted)
    print("\nHeld-out results (macro P/R/F1, macro one-vs-rest AUC)")
    print(metrics_table.round(4).to_string())
    print(f"\nWrote {len(fitted)} models, test_data.csv ({len(test)} rows) and the summary.")


if __name__ == "__main__":
    main()
