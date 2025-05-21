import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load dataset
dataset = pd.read_csv('./diabetes.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# Split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Define parameter grid for LogisticRegression
param_grid = {
    'penalty': ['l1', 'l2', 'elasticnet', None],
    'C': [0.01, 0.1, 1, 10],
    'solver': ['liblinear', 'saga'],  # 'saga' supports l1, l2, elasticnet; 'liblinear' for l1 and l2
    'l1_ratio': [None, 0.15, 0.5]      # only used if penalty='elasticnet'
}

# Initialize LogisticRegression
lr = LogisticRegression(max_iter=1000, random_state=42)

# GridSearchCV setup
grid_search = GridSearchCV(
    estimator=lr,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    n_jobs=-1,
    verbose=1
)

# Fit to training data
grid_search.fit(X_train, Y_train)

# Print all results
print("\nAll results from Grid Search (Logistic Regression):\n")
for mean_score, params in zip(grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['params']):
    print(f"Accuracy: {mean_score:.4f} for parameters: {params}")

# Best results
print("\nBest parameters found:")
print(grid_search.best_params_)
print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")
