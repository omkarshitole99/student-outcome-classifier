"""Streamlit page for the student outcome classifiers.

Loads the five saved pipelines from model/, scores whichever test CSV is
uploaded (or the bundled hold-out file) and shows the results. Nothing is
trained or refitted on this page.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # no screen on the server; figures are only rendered to images
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from charts import (
    class_f1_heat,
    confusion_figure,
    dot_plot,
    risk_histogram,
    weight_bars,
)
from prep import RAW_FEATURES, read_csv_bytes, split_features_and_target, tidy_columns
from scoring import (
    METRICS,
    confusion_counts,
    per_class_report,
    predict_with,
    score_model,
)

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "model"
SAMPLE_CSV = HERE / "test_data.csv"

# Columns shown next to a flagged student on the "Students to watch" tab.
# These are kept small on purpose; the app is for checking model behaviour, not
# for dumping the whole dataset on the screen.
CONTEXT_COLUMNS = [
    "course",
    "age_at_enrollment",
    "tuition_fees_up_to_date",
    "scholarship_holder",
    "curricular_units_2nd_sem_approved",
    "curricular_units_2nd_sem_grade",
]

PAGE_CSS = """
<style>
.block-container {max-width: 1180px; padding-top: 3.4rem;}
h1, h2, h3, h4 {font-family: Georgia, 'Times New Roman', serif !important; color: #1f2933;}
.masthead {border-left: 6px solid #0f766e; padding: 0.2rem 0 0.2rem 1rem; margin-bottom: 0.7rem;}
.masthead .title {font-family: Georgia, 'Times New Roman', serif; font-size: 1.9rem; line-height: 1.15; color: #1f2933;}
.masthead .sub {color: #52606d; margin-top: 0.3rem; font-size: 0.97rem;}
.chips {display: flex; gap: 0.6rem; flex-wrap: wrap; margin: 0.3rem 0 1.1rem 0;}
.chip {background: #efe9df; border-radius: 999px; padding: 0.2rem 0.85rem; font-size: 0.85rem; color: #3e4c59;}
.tilerow {display: grid; gap: 0.7rem; margin: 0.4rem 0 0.9rem 0;}
.tile {background: #ffffff; border: 1px solid #e3dccf; border-top: 3px solid #0f766e; border-radius: 4px; padding: 0.5rem 0.8rem 0.55rem 0.8rem;}
.tile .k {font-size: 0.7rem; letter-spacing: 0.07em; text-transform: uppercase; color: #7b8794;}
.tile .v {font-size: 1.5rem; font-weight: 600; color: #1f2933; font-variant-numeric: tabular-nums;}
table.board {border-collapse: collapse; width: 100%; font-size: 0.92rem; background: #ffffff;}
table.board th {text-align: right; padding: 0.45rem 0.7rem; border-bottom: 2px solid #1f2933; font-weight: 600;}
table.board td {text-align: right; padding: 0.42rem 0.7rem; border-bottom: 1px solid #e3dccf; font-variant-numeric: tabular-nums;}
table.board th:first-child, table.board td:first-child {text-align: left;}
table.board td.best {background: #d6ece7; font-weight: 700;}
</style>
"""


# ---------------------------------------------------------------- loading ---

@st.cache_resource(show_spinner=False)
def load_models():
    card = json.loads((MODEL_DIR / "training_summary.json").read_text(encoding="utf-8"))
    pipelines = {
        name: joblib.load(MODEL_DIR / info["file"]) for name, info in card["models"].items()
    }
    return card, pipelines


@st.cache_data(show_spinner="Scoring the saved models...")
def score_file(raw: bytes) -> dict:
    """Parse the CSV and run every saved model over it. Raises ValueError if the file is unusable."""
    card, pipelines = load_models()
    labels = card["labels"]
    table = read_csv_bytes(raw)
    features, target = split_features_and_target(table, card["feature_columns"], labels)

    rows, preds, probas, class_f1 = [], {}, {}, {}
    for name, pipeline in pipelines.items():
        predicted, proba = predict_with(pipeline, features, labels)
        preds[name], probas[name] = predicted, proba
        rows.append({"ML Model Name": name, **score_model(target, predicted, proba, labels)})
        report = per_class_report(target, predicted, labels)
        class_f1[name] = report["f1-score"].reindex(labels)
    return {
        "tidy": tidy_columns(table),
        "target": target,
        "scores": pd.DataFrame(rows).set_index("ML Model Name")[METRICS],
        "class_f1": pd.DataFrame(class_f1).T,
        "preds": preds,
        "probas": probas,
    }


# ---------------------------------------------------------------- widgets ---

def tiles(pairs) -> None:
    cells = "".join(
        f'<div class="tile"><div class="k">{html.escape(k)}</div>'
        f'<div class="v">{html.escape(v)}</div></div>'
        for k, v in pairs
    )
    style = f"grid-template-columns: repeat({len(pairs)}, 1fr);"
    st.markdown(f'<div class="tilerow" style="{style}">{cells}</div>', unsafe_allow_html=True)


def number_text(value) -> str:
    return "n/a" if pd.isna(value) else f"{value:.4f}"


def best_by_metric(scores: pd.DataFrame) -> dict:
    return {m: scores[m].idxmax() for m in METRICS if scores[m].notna().any()}


def scoreboard_html(scores: pd.DataFrame) -> str:
    best = best_by_metric(scores)
    head = "".join(f"<th>{m}</th>" for m in METRICS)
    body = []
    for name, row in scores.iterrows():
        cells = "".join(
            f'<td class="best">{number_text(row[m])}</td>'
            if best.get(m) == name
            else f"<td>{number_text(row[m])}</td>"
            for m in METRICS
        )
        body.append(f"<tr><td>{html.escape(name)}</td>{cells}</tr>")
    return (
        f'<table class="board"><thead><tr><th>ML Model Name</th>{head}</tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table>'
    )


def show(fig) -> None:
    st.pyplot(fig)
    plt.close(fig)


# ------------------------------------------------------------------- tabs ---

def tab_scoreboard(result: dict) -> None:
    scores = result["scores"]
    st.markdown(scoreboard_html(scores), unsafe_allow_html=True)
    leaders = pd.Series(best_by_metric(scores)).value_counts()
    if len(leaders):
        st.caption(
            f"Darker cells mark the best score in each column. On this file "
            f"{leaders.index[0]} is best on {leaders.iloc[0]} of {len(METRICS)} metrics."
        )
    left, right = st.columns([1.35, 1])
    with left:
        st.markdown("#### Every metric, every model")
        show(dot_plot(scores))
    with right:
        st.markdown("#### F1 for each outcome")
        show(class_f1_heat(result["class_f1"]))
        st.caption("Enrolled is the hard group: fewer students, and they sit between the other two.")


def tab_one_model(result: dict, card: dict, labels: list, chosen: str) -> None:
    info = card["models"][chosen]
    st.markdown(f"#### {chosen}")
    settings = ", ".join(f"{k} = {v}" for k, v in info["best_params"].items())
    st.caption(f"Tuned settings: {settings}.  5-fold CV macro-F1 on the training rows: {info['cv_macro_f1']:.4f}.")
    row = result["scores"].loc[chosen]
    tiles([(m, number_text(row[m])) for m in METRICS])

    counts = confusion_counts(result["target"], result["preds"][chosen], labels)
    left, right = st.columns([1, 1.1])
    with left:
        view = st.radio("Confusion matrix shows", ["Counts", "Share of each actual outcome"], horizontal=True)
        show(confusion_figure(counts, as_share=view != "Counts"))
    with right:
        st.markdown("**Classification report**")
        st.dataframe(per_class_report(result["target"], result["preds"][chosen], labels))

    if info.get("top_features"):
        signed = chosen == "Logistic Regression"
        st.markdown("#### What this model leans on")
        show(weight_bars(info["top_features"], signed))
        st.caption(
            "Red pushes towards Dropout, teal pushes away (Dropout row of the coefficients)."
            if signed
            else "Share of the tree's splitting gain carried by each input."
        )


def tab_watchlist(result: dict, labels: list, chosen: str) -> None:
    st.markdown(f"Students ranked by the chance of dropping out according to **{chosen}**.")
    threshold = st.slider("Flag a student when the predicted chance is at least", 0.10, 0.90, 0.50, 0.05)

    p_dropout = result["probas"][chosen][:, labels.index("Dropout")]
    actual = result["target"].to_numpy()
    is_dropout = actual == "Dropout"
    flagged = p_dropout >= threshold
    hits = int((flagged & is_dropout).sum())
    tiles([
        ("Students flagged", f"{int(flagged.sum())}"),
        ("Flagged who did drop out", "n/a" if not flagged.any() else f"{hits / flagged.sum():.1%}"),
        ("Dropouts caught", "n/a" if not is_dropout.any() else f"{hits / is_dropout.sum():.1%}"),
        ("Dropouts in the file", f"{int(is_dropout.sum())}"),
    ])
    show(risk_histogram(p_dropout, is_dropout, threshold))

    view = result["tidy"][CONTEXT_COLUMNS].copy()
    view.insert(0, "row_in_file", np.arange(1, len(view) + 1))
    view["p_dropout"] = p_dropout.round(3)
    view["predicted"] = result["preds"][chosen]
    view["actual"] = actual
    view = view[flagged].sort_values("p_dropout", ascending=False)
    if view.empty:
        st.info("Nobody reaches that threshold. Lower the slider.")
        return
    st.dataframe(view.head(300), hide_index=True)
    st.download_button(
        "Download the flagged list (CSV)",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name="flagged_students.csv",
        mime="text/csv",
    )


def tab_method(card: dict) -> None:
    data, split = card["dataset"], card["split"]
    st.markdown(
        f"""
**Data.** {data['source']}. {data['rows']:,} students, {data['raw_features']} input columns,
three outcomes. The columns describe the student at enrolment plus the first two semesters,
so a prediction here is made with first-year results in hand, not on the day of enrolment.

**Split.** {split['train_rows']:,} rows for fitting and tuning, {split['test_rows']:,} held back
(stratified, seed {split['seed']}). The held-back rows are `test_data.csv`.

**Preparation (inside every saved model).** Code-like integers (course, parents' jobs and
qualifications, nationality, application mode, marital status) are one-hot encoded, with codes
seen in under 1% of rows pooled together. Grades, unit counts and rates are standardised.
Yes/no columns are passed through. Three columns were added: the share of enrolled units passed
in each semester, and the change in average grade between the semesters.

**Tuning.** Grid search with {card['tuning']['folds']}-fold stratified cross-validation on the
fitting rows only, scored by macro-F1. The SVM is wrapped in a sigmoid calibrator afterwards so it
can output probabilities for AUC.

**Metrics.** Precision, recall and F1 are macro averages. AUC is the macro one-vs-rest AUC.
Accuracy and MCC are the plain multi-class forms.
"""
    )
    tuned = pd.DataFrame(
        [
            {
                "Model": name,
                "Tuned settings": ", ".join(f"{k}={v}" for k, v in info["best_params"].items()),
                "Settings tried": info["grid_size"],
                "CV macro-F1": info["cv_macro_f1"],
            }
            for name, info in card["models"].items()
        ]
    ).set_index("Model")
    st.dataframe(tuned)
    mix = pd.DataFrame(
        {
            "All rows": data["class_counts"],
            "Fitting rows": split["train_class_counts"],
            "Held back": split["test_class_counts"],
        }
    )
    st.markdown("**Outcome counts**")
    st.dataframe(mix)
    st.caption(
        "Source: Realinho, Vieira Martins, Machado and Baptista (2021), Predict Students' Dropout and "
        "Academic Success, UCI Machine Learning Repository, doi 10.24432/C5MC89."
    )


# ------------------------------------------------------------------- page ---

def masthead(subtitle: str, chips) -> None:
    chip_html = "".join(f'<span class="chip">{html.escape(c)}</span>' for c in chips)
    st.markdown(
        f'<div class="masthead"><div class="title">Student outcome classifiers</div>'
        f'<div class="sub">{html.escape(subtitle)}</div></div>'
        f'<div class="chips">{chip_html}</div>',
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Student outcome classifiers", layout="wide")
    st.markdown(PAGE_CSS, unsafe_allow_html=True)
    card, pipelines = load_models()
    labels = card["labels"]

    with st.sidebar:
        st.markdown("### Test file")
        upload = st.file_uploader(
            f"CSV with the {len(RAW_FEATURES)} input columns and the target column", type=["csv"]
        )
        st.caption(
            "Using the bundled hold-out file until you upload one."
            if upload is None
            else f"Scoring {upload.name}."
        )
        st.caption(
            "Upload only rows the models were not fitted on. The full UCI file contains "
            "the fitting rows, so its scores come out too high."
        )
        st.markdown("### Model")
        # This dropdown is included for the assignment requirement. The
        # scoreboard still compares all models, while this choice controls the
        # detailed confusion matrix, report and dropout watch-list.
        chosen = st.selectbox("Model to look at closely", list(pipelines))
        st.caption("The scoreboard always compares all five.")

    raw = upload.getvalue() if upload is not None else SAMPLE_CSV.read_bytes()
    source = upload.name if upload is not None else "test_data.csv (bundled)"
    subtitle = (
        "Dropout, still enrolled or graduate, from first-year records. "
        "Five saved models scored on a test file; nothing is retrained here."
    )
    try:
        result = score_file(raw)
    except ValueError as problem:
        masthead(subtitle, [source])
        st.error(str(problem))
        st.stop()
    except Exception as problem:  # unreadable bytes, odd delimiter, and so on
        masthead(subtitle, [source])
        st.error(f"That file could not be read: {problem}")
        st.stop()

    masthead(
        subtitle,
        [source, f"{len(result['target']):,} students", f"{len(RAW_FEATURES)} input columns", f"{len(labels)} outcomes"],
    )
    scoreboard, closer, watch, method = st.tabs(
        ["Scoreboard", "One model closely", "Students to watch", "Data and method"]
    )
    with scoreboard:
        tab_scoreboard(result)
    with closer:
        tab_one_model(result, card, labels, chosen)
    with watch:
        tab_watchlist(result, labels, chosen)
    with method:
        tab_method(card)


main()
