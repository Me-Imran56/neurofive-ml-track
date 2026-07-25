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

**Dataset:** Titanic - Machine Learning from Disaster (891 rows, 12 columns). `titanic.csv`
included for convenience — swap in your own Kaggle download if preferred (same columns).

### To run locally
```
pip install pandas numpy matplotlib seaborn jupyter
jupyter notebook titanic_eda.ipynb
```

### Key findings
- 891 rows, 12 columns; missing: Age (177), Cabin (687), Embarked (2)
- Women survived ~74% vs. ~19% for men — Sex is the strongest predictor of survival
- 1st class passengers paid dramatically more and survived at a higher rate than 3rd class
