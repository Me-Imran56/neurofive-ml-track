# neurofive-ml-track

## Task 1: Environment Setup + First EDA (Titanic Dataset)
Loaded the Titanic dataset with pandas, inspected it with `.info()`, `.describe()`, `.head()`,
identified missing values and column types, and wrote a short data-story summary.

## Task 2: Data Cleaning + Visual EDA
- Handled missing values: `Age` (median fillna), `Embarked` (mode fillna), `Cabin`
  (converted to a `Has_Cabin` binary flag instead of imputing — too sparse at ~77% missing).
- Detected Fare outliers using IQR + boxplot.
- Built 4 visualizations: Age histogram, Fare-by-class boxplot, Sex survival-rate bar chart,
  and a full correlation heatmap.
- Answered: which feature most affects survival → **Sex**, followed by **Pclass**.

## Task 3: First ML Model — Logistic Regression
**Approach:**
- Dropped non-predictive columns (`Name`, `Ticket`, `PassengerId`).
- One-hot encoded `Sex` and `Embarked` with `pd.get_dummies(drop_first=True)`.
- Split data 80/20 with `train_test_split`, using `stratify=y` to preserve the survival
  ratio in both sets.
- Trained a `LogisticRegression` model (`sklearn.linear_model`) on the training set.
- Evaluated with `accuracy_score` and a confusion matrix.

**Result: 80.45% test accuracy.**

Confusion matrix (see `confusion_matrix.png` / notebook):
| | Predicted: Did not survive | Predicted: Survived |
|---|---|---|
| **Actual: Did not survive** | 96 (TN) | 14 (FP) |
| **Actual: Survived** | 21 (FN) | 48 (TP) |

The model is noticeably better at correctly identifying passengers who did **not** survive
(96/110 ≈ 87%) than those who did (48/69 ≈ 70%), reflecting the class imbalance in the
training data (~38% of passengers survived overall).

---

**Dataset:** Titanic - Machine Learning from Disaster (891 rows, 12 columns). `titanic.csv`
included for convenience — swap in your own Kaggle download if preferred (same columns).

### To run locally
```
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook titanic_eda.ipynb
```

## Task 4: Regression — California Housing Prices
File: `housing_regression.ipynb` (separate notebook — new dataset/topic from Tasks 1-3).

**Approach:**
- Dataset: California Housing (20,640 districts, 1990 census) — `housing.csv`, the standard
  public mirror of the sklearn/Kaggle-equivalent housing regression dataset (used in place of
  the deprecated Boston Housing dataset).
- Filled 207 missing `total_bedrooms` values with the median.
- Selected 5 features believed to most affect price: `median_income`, `total_rooms`,
  `housing_median_age`, `latitude`, `longitude`.
- Trained a `LinearRegression` model (`sklearn.linear_model`), 80/20 train-test split.
- Evaluated with RMSE and R².

**Result: RMSE = $73,793 | R² = 0.58**

R² of 0.58 means the model explains about 58% of the variation in house prices across
districts using just these 5 features — a reasonable ballpark estimate, but not precise
enough to replace a real appraisal (see notebook for the full plain-English explanation and
the predicted-vs-actual scatter plot).

### To run Task 4 locally
```
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook housing_regression.ipynb
```

## Task 5: Model Evaluation Deep-Dive + Hyperparameter Tuning
File: `titanic_eda.ipynb` (continues after Task 3).

**Approach:**
- Calculated Precision, Recall, and F1-score with `classification_report` on the Task 3
  Logistic Regression model.
- Explained why accuracy alone is misleading on imbalanced data (Titanic is ~62%/38% split) —
  a model that always predicts the majority class can still look accurate while learning
  nothing useful; Precision/Recall/F1 expose that in a way accuracy hides.
- Tuned 2 hyperparameters — `C` (regularization strength) and `solver` — with `GridSearchCV`
  (5-fold cross-validation, scored on F1).

**Before vs. After tuning (test set):**
| Metric | Before Tuning | After Tuning | Change |
|---|---|---|---|
| Accuracy | 0.8045 | 0.8045 | 0.0 |
| Precision (Survived) | 0.7742 | 0.7742 | 0.0 |
| Recall (Survived) | 0.6957 | 0.6957 | 0.0 |
| F1-score (Survived) | 0.7328 | 0.7328 | 0.0 |

**Result: no change.** `GridSearchCV` found `C=1, solver=lbfgs` was best — which is exactly
scikit-learn's default. On a simple linear model with a small, clean dataset, the defaults
were already near-optimal for this feature set; tuning tends to matter more on models with
more capacity to overfit/underfit (tree-based models, SVMs, gradient boosting). The exercise's
value here was confirming this systematically via cross-validation rather than guessing.

## Task 6: Customer Churn Prediction — Telco Dataset
File: `churn_prediction.ipynb` (new dataset/topic).

**Approach:**
- Dataset: IBM Telco Customer Churn (7,043 customers, 21 columns) — the original public
  source of the Kaggle "Telco Customer Churn" dataset.
- Quick EDA: month-to-month contracts churn at 42.7% vs. 11.3% (one year) and 2.8% (two year);
  churners also skew toward lower tenure and higher monthly charges.
- Cleaned `TotalCharges` (11 blank entries from tenure=0 customers, converted and filled as 0).
- One-hot encoded all categorical columns.
- **Class imbalance noted:** ~73% no-churn vs. ~27% churn. Not resampled/rebalanced in this
  task, but called out explicitly, and evaluation uses Precision/Recall/F1 (not just
  accuracy) on the Churn class specifically for that reason.
- Trained and compared **Logistic Regression** vs. **Decision Tree** (`max_depth=5`).

**Model comparison (test set):**
| Model | Accuracy | Precision (Churn) | Recall (Churn) | F1 (Churn) |
|---|---|---|---|---|
| Logistic Regression | 0.8062 | 0.66 | 0.56 | 0.60 |
| Decision Tree | 0.7942 | 0.63 | 0.54 | 0.58 |

**Top 3 features driving churn (Decision Tree `.feature_importances_`):**
1. `tenure` (0.42)
2. `InternetService_Fiber optic` (0.36)
3. `TotalCharges` (0.04)

**Business summary:** newer customers (low tenure) are by far the highest churn risk, so the
first few months of the relationship are the critical retention window. Fiber optic internet
customers churn notably more than other segments — worth investigating pricing/reliability
there directly. Logistic Regression slightly outperforms the Decision Tree numerically, but
the Tree is easier to explain to non-technical stakeholders since its decision rules can be
traced directly.

### To run Task 6 locally
```
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook churn_prediction.ipynb
```

## Task 7: Production-Style Pipeline + Feature Engineering
File: `titanic_eda.ipynb` (continues after Task 5) + `titanic_pipeline.joblib` (saved model).

**Approach:**
- Engineered 2 new features: `FamilySize` (SibSp + Parch + 1) and `IsAlone` (1 if travelling
  solo). Both show a real survival pattern — solo travellers and very large families (5+)
  survived less often.
- Built a single `Pipeline` with a `ColumnTransformer`: `StandardScaler` on numerical columns
  (`Age`, `Fare`, `SibSp`, `Parch`, `FamilySize`, `IsAlone`, `Has_Cabin`) and `OneHotEncoder`
  on categorical columns (`Sex`, `Embarked`, `Pclass`), feeding into `LogisticRegression`.
- Fit and evaluated the pipeline against the Task 3 manual approach.
- Saved the final fitted pipeline with `joblib.dump()` and verified the reloaded model
  predicts identically.

**Pipeline vs. manual comparison:**
| Approach | Accuracy | F1-score |
|---|---|---|
| Manual (Task 3) | 0.8045 | 0.7328 |
| Pipeline + Feature Engineering | 0.8045 | 0.7328 |

**Result: identical performance.** The pipeline correctly reproduces the manual preprocessing
(same accuracy/F1), confirming no information was lost in the `ColumnTransformer` setup. The
engineered features didn't improve this particular Logistic Regression result — likely
because `FamilySize` is a linear combination of `SibSp`/`Parch`, which the model already had
access to — but they didn't hurt performance either, and could help more with a model that
captures feature interactions differently (e.g. a Decision Tree).

**Why a pipeline matters:** it bundles preprocessing and modeling into one object, so
`pipeline.fit()` only fits the scaler/encoder on training data (no leakage) and
`pipeline.predict()` on new raw data automatically applies the exact same transformations —
no manual repetition, and the whole thing is a single object you can save and deploy.

### To run Task 7 locally
```
pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter
jupyter notebook titanic_eda.ipynb
# Reload the saved pipeline elsewhere with:
# import joblib; pipeline = joblib.load('titanic_pipeline.joblib')
```

## Task 8: Ensemble Methods — Random Forest & XGBoost
File: `titanic_eda.ipynb` (continues after Task 7).

**Approach:**
- Trained `RandomForestClassifier` (n_estimators=200, max_depth=6) and `XGBClassifier`
  (n_estimators=200, max_depth=4, learning_rate=0.1) on the same engineered feature set from
  Task 7 (`FamilySize`, `IsAlone`, `Has_Cabin`, etc.), one-hot encoded.
- Compared both against the Task 3 Logistic Regression baseline.
- Plotted and compared feature importances for both ensembles.

**Model comparison:**
| Model | Accuracy | F1-score |
|---|---|---|
| Logistic Regression (Task 3) | 0.8045 | 0.7328 |
| Random Forest | 0.7933 | 0.6992 |
| XGBoost | 0.7821 | 0.6977 |

**Result: Logistic Regression actually outperformed both ensembles on this dataset.** This is
reported honestly rather than forced — Titanic is small (~891 rows) with a fairly linear
relationship between key features (Sex, Pclass) and survival, and both ensembles used modest,
untuned hyperparameters. Ensembles typically show their real advantage on larger, more
non-linear datasets (e.g. the Telco churn data from Task 6).

**Top 3 features:**
- Random Forest: `Sex_male`, `Fare`, `Age`
- XGBoost: `Sex_male`, `Pclass_3`, `Has_Cabin`

Both agree `Sex_male` is by far the most important feature, consistent with earlier EDA
findings; they diverge on the next-ranked features (Fare/Age vs. Pclass/Has_Cabin), both
plausible proxies for socioeconomic status.

**Random Forest vs. XGBoost — how they differ:** Random Forest builds many decision trees
independently in parallel (bagging) on random subsets of rows/features and averages their
votes, which reduces variance and overfitting. XGBoost builds trees sequentially (boosting),
where each new tree specifically corrects the errors of the trees before it, refining the
model step by step with built-in regularization. Boosting methods like XGBoost often edge out
bagging methods like Random Forest on structured/tabular data, though not universally, as
seen here.

### To run Task 8 locally
```
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib jupyter
jupyter notebook titanic_eda.ipynb
```

## Task 9: Handling Class Imbalance — SMOTE
File: `churn_prediction.ipynb` (continues after the Task 6 churn model).

**Approach:**
- Reused the Telco churn dataset (fraud dataset was too large/Kaggle-only, as the task
  description allowed).
- Visualized class balance: **5,174 No Churn vs. 1,869 Churn (~73%/27%)**.
- Explained in writing why accuracy alone is misleading here — a model that always predicts
  "No Churn" would score ~73% accuracy while catching zero actual churners.
- Applied **SMOTE** (`imbalanced-learn`) to the training set only (never the test set, to
  avoid data leakage), balancing it from 4,139/1,495 to an even 4,139/4,139.
- Retrained Logistic Regression on the SMOTE-balanced data and compared before/after.

**Before vs. after SMOTE (test set):**
| Metric | Before SMOTE | After SMOTE | Change |
|---|---|---|---|
| Accuracy | 0.8062 | 0.7601 | −0.0461 |
| Precision (Churn) | 0.6593 | 0.5411 | −0.1182 |
| Recall (Churn) | 0.5588 | 0.6337 | **+0.0749** |
| F1-score (Churn) | 0.6049 | 0.5837 | −0.0212 |

**Result: a genuine precision/recall trade-off.** SMOTE improved Recall (catches more real
churners: 63% vs. 56%) but reduced Precision, Accuracy, and F1 — more false alarms in
exchange for missing fewer real churners. This is reported honestly as a trade-off, not a
clean win: which model is "better" depends on the real business cost of a missed churner vs.
a wasted retention offer, a decision accuracy alone would have completely hidden.

### To run Task 9 locally
```
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter
jupyter notebook churn_prediction.ipynb
```

## Task 10: Streamlit Web App — Live Deployment
Files: `app.py`, `requirements.txt`, `titanic_pipeline.joblib` (the Task 7 saved pipeline).

**Approach:**
- Used the best-saved model: the Task 7 Logistic Regression pipeline (80.45% test accuracy,
  bundled with `StandardScaler` + `OneHotEncoder` preprocessing — no manual encoding needed
  in the app itself).
- Built a Streamlit app with input fields for Passenger Class, Sex, Age, Fare, Siblings/
  Spouses aboard, Parents/Children aboard, Port of Embarkation, and whether a cabin was
  recorded — the app computes `FamilySize`/`IsAlone` automatically from the family inputs.
- A **Predict** button runs `pipeline.predict()` and `pipeline.predict_proba()`, showing the
  prediction plus confidence percentage.
- Tested locally (`streamlit run app.py`) and verified predictions make sense: a 1st-class
  woman → 93% predicted survival; a 3rd-class man → 6% predicted survival.

**Live app:** _https://neurofive-ml-track-u3fewb7saemdxguwsupx4r.streamlit.app/_


### To run locally
```
pip install streamlit scikit-learn pandas joblib
streamlit run app.py
```
