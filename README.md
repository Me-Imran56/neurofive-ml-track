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
