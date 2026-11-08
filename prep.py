"""Column cleanup, feature groups and the shared preprocessing step.

The raw UCI export has a couple of awkward headers (one ends in a tab,
'Nacionality' is misspelt), so every table goes through tidy_columns()
first. Training and the Streamlit page both import this file, which keeps
the two sides agreeing on column names.
"""

from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "target"

# Column groups were made after reading the UCI attribute list and checking
# the raw values. These integer columns are really category codes (course,
# occupation, qualification, ...), not measurements. For example, course 9500
# is not "more" than course 171, so these get one-hot encoded.
CODE_COLUMNS = [
    "marital_status",
    "application_mode",
    "course",
    "previous_qualification",
    "nationality",
    "mothers_qualification",
    "fathers_qualification",
    "mothers_occupation",
    "fathers_occupation",
]

# Yes/no answers already stored as 0/1 in the UCI file, so no one-hot encoding
# is needed for them.
FLAG_COLUMNS = [
    "daytime_evening_attendance",
    "displaced",
    "educational_special_needs",
    "debtor",
    "tuition_fees_up_to_date",
    "gender",
    "scholarship_holder",
    "international",
]

# Real quantities: grades, ages, unit counts and the economic indicators.
# These are scaled inside the pipeline because LR, KNN and SVM depend on the
# size of the input values.
NUMBER_COLUMNS = [
    "application_order",
    "previous_qualification_grade",
    "admission_grade",
    "age_at_enrollment",
    "curricular_units_1st_sem_credited",
    "curricular_units_1st_sem_enrolled",
    "curricular_units_1st_sem_evaluations",
    "curricular_units_1st_sem_approved",
    "curricular_units_1st_sem_grade",
    "curricular_units_1st_sem_without_evaluations",
    "curricular_units_2nd_sem_credited",
    "curricular_units_2nd_sem_enrolled",
    "curricular_units_2nd_sem_evaluations",
    "curricular_units_2nd_sem_approved",
    "curricular_units_2nd_sem_grade",
    "curricular_units_2nd_sem_without_evaluations",
    "unemployment_rate",
    "inflation_rate",
    "gdp",
]

RAW_FEATURES = CODE_COLUMNS + FLAG_COLUMNS + NUMBER_COLUMNS

# Columns built from the raw ones by add_progress_features(). They are not in
# the original UCI file; they were added for this assignment because semester
# progress is easier to compare as a rate than as only raw counts.
EXTRA_COLUMNS = ["pass_rate_1st_sem", "pass_rate_2nd_sem", "grade_change"]


def tidy_name(raw_name) -> str:
    """'Mother's qualification' -> 'mothers_qualification', and so on."""
    text = str(raw_name).strip().lower().replace("'", "")
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return {"nacionality": "nationality"}.get(text, text)


def tidy_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [tidy_name(name) for name in out.columns]
    if TARGET in out.columns:
        out[TARGET] = out[TARGET].astype(str).str.strip()
    return out


def add_progress_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Share of enrolled units passed in each semester, and the grade change.

    A student who enrolled in nothing gets a pass rate of 0 instead of a
    divide-by-zero.
    """
    out = frame.copy()
    for sem in ("1st", "2nd"):
        enrolled = out[f"curricular_units_{sem}_sem_enrolled"].astype(float)
        approved = out[f"curricular_units_{sem}_sem_approved"].astype(float)
        out[f"pass_rate_{sem}_sem"] = (approved / enrolled.where(enrolled > 0)).fillna(0.0)
    out["grade_change"] = (
        out["curricular_units_2nd_sem_grade"].astype(float)
        - out["curricular_units_1st_sem_grade"].astype(float)
    )
    return out


def input_columns(use_extra: bool = True) -> list[str]:
    """Columns the saved pipelines expect, in order."""
    return RAW_FEATURES + (EXTRA_COLUMNS if use_extra else [])


def build_preprocessor(use_extra: bool = True) -> ColumnTransformer:
    numbers = NUMBER_COLUMNS + (EXTRA_COLUMNS if use_extra else [])
    return ColumnTransformer(
        transformers=[
            (
                "codes",
                # Codes seen in under 1% of the training rows are pooled into one
                # bucket; a code that never appeared in training lands there too.
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=0.01,
                    sparse_output=False,
                ),
                CODE_COLUMNS,
            ),
            ("numbers", StandardScaler(), numbers),
            ("flags", "passthrough", FLAG_COLUMNS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def read_csv_bytes(raw: bytes) -> pd.DataFrame:
    """Parse an uploaded CSV. Comma and semicolon separated files both work."""
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    if not text.strip():
        raise ValueError("The file is empty.")
    header = text.splitlines()[0]
    sep = ";" if header.count(";") > header.count(",") else ","
    return pd.read_csv(io.StringIO(text), sep=sep)


def split_features_and_target(table: pd.DataFrame, feature_columns, known_labels):
    """Validate an uploaded table and return (model input frame, target).

    The file needs every raw feature plus the target column. Anything missing
    or non-numeric is reported instead of guessed.
    """
    table = tidy_columns(table)
    missing = [name for name in RAW_FEATURES + [TARGET] if name not in table.columns]
    if missing:
        raise ValueError("These columns are missing from the CSV: " + ", ".join(missing))

    numbers = table[RAW_FEATURES].apply(pd.to_numeric, errors="coerce")
    bad_cells = int(numbers.isna().sum().sum())
    if bad_cells:
        raise ValueError(
            f"{bad_cells} cells in the feature columns are empty or not numbers."
        )

    target = table[TARGET]
    if target.str.lower().isin(["", "nan"]).any():
        raise ValueError("The target column has empty cells.")
    unknown = sorted(set(target) - set(known_labels))
    if unknown:
        raise ValueError(
            "Unexpected values in the target column: "
            + ", ".join(unknown)
            + ". Expected: "
            + ", ".join(known_labels)
        )

    return add_progress_features(numbers)[list(feature_columns)], target
