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
