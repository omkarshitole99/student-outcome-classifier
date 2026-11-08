# Student outcome classifiers

Machine Learning, Assignment 2 (M.Tech AIML, BITS WILP).

Five classifiers trained on the UCI "Predict Students' Dropout and Academic Success" data, saved with joblib, and served through a Streamlit page that scores an uploaded test CSV.

## a. Problem statement

A university wants to tell, from first-year records, which students will drop out, which will still be enrolled when the course is due to end, and which will graduate. Each student is described by enrolment details, family background, economic indicators and results from the first two semesters. The task is to predict one of three outcomes: Dropout, Enrolled or Graduate.

The classes are unequal (Graduate 49.9%, Dropout 32.1%, Enrolled 17.9%), so accuracy alone can look fine while the Enrolled group is missed. Precision, recall and F1 are therefore macro averages. All five models share one train/test split, are tuned by cross-validation on the training rows only, and are scored once on the held-out rows. Always answering "Graduate" would give 0.4994 accuracy on the held-back rows, which is the bar to beat.

## b. Dataset description

| Item | Detail |
|---|---|
| Source | UCI Machine Learning Repository, dataset 697. Realinho, Vieira Martins, Machado and Baptista (2021), doi 10.24432/C5MC89 |
| Students | 4,424 |
| Input columns | 36 |
| Target | `Target`: Graduate 2,209, Dropout 1,421, Enrolled 794 |
| Missing values | none, and no duplicate rows |

The dataset itself is from UCI. The model families and metric names are from the assignment PDF. The preprocessing choices, feature-rate columns, model grids and Streamlit layout were made for this dataset rather than copied from an existing repository.

The 36 inputs fall into three groups:

- 9 integer codes that stand for categories: marital status, application mode, course, previous qualification, nationality, mother's and father's qualification, mother's and father's occupation.
- 8 yes/no columns stored as 0/1: daytime/evening attendance, displaced, educational special needs, debtor, tuition fees up to date, gender, scholarship holder, international.
- 19 numeric columns: application order, previous qualification grade, admission grade, age at enrolment, the credited / enrolled / evaluated / approved / grade / without-evaluation figures for each of the first two semesters, and the unemployment rate, inflation rate and GDP.

Points that shaped the preparation:

- The code columns are labels, not amounts (course 9500 is not "more" than course 171), so they are one-hot encoded. Codes seen in under 1% of the training rows are pooled into one bucket, which also catches codes that never appeared in training.
- The semester columns mean a prediction is made after the second semester, not on the day of enrolment.
- Unemployment, inflation and GDP are the same for every student of a given period, so they say little about an individual.
- One header in the raw file ends in a tab and "Nacionality" is misspelt. `prep.py` turns every header into snake_case and fixes the spelling. The app applies the same function, so a file with the original UCI headers is accepted as well.
- Three columns are added to the raw ones: the share of enrolled units passed in the 1st and 2nd semester (0 when nothing was enrolled) and the change in average grade between the semesters. Five-fold macro-F1 on the training rows with and without them: Logistic Regression 0.7099 to 0.7105, SVM 0.7030 to 0.7077. The gain is small, and the columns were kept. The Decision Tree puts 77% of its importance on the 2nd-semester pass rate.

Split: 80/20, stratified, `random_state=5278`.

| | Rows | Graduate | Dropout | Enrolled |
|---|---:|---:|---:|---:|
| Fitting and tuning | 3,539 | 1,767 | 1,137 | 635 |
| Held back (`test_data.csv`) | 885 | 442 | 284 | 159 |

`test_data.csv` holds the 36 original input columns and `Target` for the held-back students. The three added columns are computed from those, in `prep.py`, by both the training script and the app.

## c. Github Repository Link

https://github.com/omkarshitole99/student-outcome-classifier

Live app: https://2026ak05278-student-outcome-classifier.streamlit.app/

```
student-outcome-classifier/
  app.py                       Streamlit page
  charts.py                    the figures used by the page and the notebook
  prep.py                      column cleanup, feature groups, preprocessing
  scoring.py                   the six metrics, confusion matrix, report
  train_models.py              tuning, hold-out scoring, saving
  model_training.ipynb         the same steps as train_models.py, one cell at a time, with outputs
  requirements.txt
  README.md
  test_data.csv                885 held-back students (inputs and Target)
  data/uci_students_dropout.csv   the UCI file, unchanged
  model/
    logistic_regression.joblib
    decision_tree.joblib
    knn.joblib
    naive_bayes.joblib
    svm.joblib
    training_summary.json      settings, CV scores, hold-out scores, feature weights
    holdout_metrics.csv
  .streamlit/config.toml       theme
```

Preprocessing artifacts: the fitted encoder and scaler are stored inside each `.joblib` file (the `prep` step of the pipeline), so a saved model takes the raw columns directly. `model/training_summary.json` records the input columns and the outcome labels the app expects.

## d. Models used

Each saved file is a scikit-learn pipeline: the shared preprocessing (one-hot codes, standardised numbers, 0/1 flags passed through) followed by the classifier. Settings were chosen by grid search with 5-fold stratified cross-validation on the 3,539 fitting rows, scored by macro-F1.

| Model | Searched | Chosen |
|---|---|---|
| Logistic Regression | C 0.03 to 3; class weight none or balanced | C = 0.1, balanced |
| Decision Tree | depth 3 to 10; min leaf 5 to 40; gini or entropy; class weight none or balanced | depth 4, min leaf 10, gini, no weights |
| KNN | k 5 to 35; uniform or distance weights; Manhattan or Euclidean | k = 15, uniform, Manhattan |
| Naive Bayes (Gaussian) | var_smoothing 1e-9 to 3 | 0.1 |
| Support Vector Machine | C 1 to 100; gamma "scale" or 0.0003 to 0.01; class weight none or balanced | RBF, C = 100, gamma = 0.001, balanced |

Two first-pass grids had their best value on an edge (Naive Bayes at the top, SVM at the top of C and bottom of gamma), so those grids were moved and the search was repeated. A spot check beyond the SVM grid (C = 300 and 1000) did not beat the chosen setting. The SVM is wrapped in a sigmoid calibrator after tuning so it can output probabilities for AUC.

### Comparison table

Held-back rows (885). Precision, recall and F1 are macro averages, AUC is the macro one-vs-rest AUC, accuracy and MCC are the plain multi-class forms.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7548 | 0.8814 | 0.7117 | 0.7113 | 0.7095 | 0.6054 |
| Decision Tree | 0.7243 | 0.8410 | 0.6791 | 0.6561 | 0.6632 | 0.5492 |
| KNN | 0.7412 | 0.8519 | 0.7123 | 0.6285 | 0.6418 | 0.5730 |
| Naive Bayes | 0.7028 | 0.8126 | 0.6561 | 0.6211 | 0.6295 | 0.5099 |
| Support Vector Machine | 0.7831 | 0.8582 | 0.7403 | 0.7201 | 0.7283 | 0.6428 |

F1 for each outcome on the same rows:

| ML Model Name | Dropout | Enrolled | Graduate |
|---|---:|---:|---:|
| Logistic Regression | 0.7918 | 0.5072 | 0.8294 |
| Decision Tree | 0.7623 | 0.4159 | 0.8116 |
| KNN | 0.7594 | 0.3448 | 0.8213 |
| Naive Bayes | 0.6749 | 0.4013 | 0.8123 |
| Support Vector Machine | 0.8074 | 0.5249 | 0.8525 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Second on accuracy, F1 and MCC, and best on AUC (0.8814). Balanced class weights with a fairly strong penalty (C = 0.1) worked best. Its largest coefficient is `tuition_fees_up_to_date`, which pulls hard away from Dropout, followed by approved units in both semesters. It finds 55% of the Enrolled students, but only 47% of the students it labels Enrolled really are. |
| Decision Tree | Depth 4 was chosen, so the tree stays small and readable, and 77% of its importance sits on the 2nd-semester pass rate. It has the biggest drop from cross-validation (0.692) to the held-back rows (0.663) of the five, which fits a single tree being unstable. Enrolled F1 is 0.416. |
| KNN | Leans towards the big class: it finds 96% of Graduates but only 25% of Enrolled students, and 101 of the 159 Enrolled students were labelled Graduate. Macro precision is good (0.7123) but macro recall is low (0.6285), so F1 falls to 0.6418. Many one-hot columns probably add noise to the distances. |
| Naive Bayes | Lowest on all six metrics. The independence assumption does not hold here: units enrolled, evaluated and approved in a semester move together, and so do the two semesters, and the one-hot columns are not Gaussian. It needed heavy smoothing (0.1) and finds only 58% of dropouts. |
| Support Vector Machine | Best on accuracy, precision, recall, F1 and MCC, and best F1 in every one of the three outcomes (Enrolled 0.5249). The RBF kernel with a large C and small gamma suits the overlapping groups. Its AUC (0.8582) is below Logistic Regression's because its probabilities come from calibrating SVM scores rather than from the model itself. |
| Overall, Winner for your dataset? | Support Vector Machine, ahead on five of six metrics. The lead over Logistic Regression is small (F1 0.7283 against 0.7095, and fold-to-fold F1 varies by about 0.017), so a different split could narrow or flip it. Logistic Regression is the simpler model and has the better AUC. Every model beats the 0.4994 baseline, and every model is weakest on Enrolled (F1 from 0.34 to 0.52). |

## Run it

Built and tested on Python 3.14 with the versions that `requirements.txt` resolves to.

```
pip install -r requirements.txt
streamlit run app.py
```

To rebuild the split, the models and `model/` from the UCI file (same seed, same numbers on the same library versions):

```
python train_models.py
```

`model_training.ipynb` runs the same steps one cell at a time and shows the output of each: data checks, a few plots, the tuning result and confusion matrix for every model, the comparison table, saving, and a reload check. Opening it needs Jupyter, which is not in `requirements.txt` because the app does not use it.

The page has four tabs: a scoreboard comparing all five models on the uploaded file, a closer look at the model chosen in the sidebar (metrics, confusion matrix, classification report), a list of students the chosen model flags as likely dropouts at an adjustable threshold, and a short description of the data and method. It loads the saved models and never trains.

The uploaded CSV needs the 36 input columns and `Target` (extra columns are ignored, column order does not matter). Upload rows the models were not fitted on, such as `test_data.csv`. The full UCI file contains the fitting rows, so its scores come out too high.

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Open https://streamlit.io/cloud and sign in with GitHub.
3. New app, pick the repository and branch (main), set the main file to `app.py`.
4. Deploy, then paste the live URL above.

The model files were saved with scikit-learn 1.9.0, which is why `requirements.txt` pins it.
