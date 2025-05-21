# imports
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score
sc = StandardScaler()

# dataset splitting
dataset = pd.read_csv('./data/diabetes.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)


# List of columns to replace 0s
cols_with_zero_invalid = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Replace 0s with median values
for col in cols_with_zero_invalid:
    dataset[col] = dataset[col].replace(0, dataset[col].median())

# feature scaling
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# logistic regression
lreg = LogisticRegression(C=0.1, l1_ratio=0.5, penalty="elasticnet", solver="saga", random_state=42)
lreg.fit(X_train, Y_train)
y_pred_lreg= lreg.predict(X_test)



# random forest
rf = RandomForestClassifier(class_weight="balanced", max_depth=5, max_features='sqrt', min_samples_leaf=2, min_samples_split=2, n_estimators=100, random_state=42)
rf.fit(X_train, Y_train)
y_pred_rf = rf.predict(X_test)


# decision tree
dt = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=3, max_features=None, min_samples_leaf=4, min_samples_split=2, random_state=42)
dt.fit(X_train, Y_train)
y_pred_dt = dt.predict(X_test)



# knn
knn = KNeighborsClassifier(metric="euclidean", n_neighbors=11, p=1, weights="distance")
knn.fit(X_train, Y_train)
y_pred_knn = knn.predict(X_test)



# confusion matrix
print("Logistic Regression")
cm1 = confusion_matrix(Y_test, lreg.predict(X_test))
print(cm1)
print(accuracy_score(Y_test, lreg.predict(X_test)))

print("Random Forest Classification")
cm2 = confusion_matrix(Y_test, rf.predict(X_test))
print(cm2)
print(accuracy_score(Y_test, rf.predict(X_test)))

print("Decision Tree Classification")
cm3 = confusion_matrix(Y_test, dt.predict(X_test))
print(cm3)
print(accuracy_score(Y_test, dt.predict(X_test)))

print("K Nearest Neighbors Classification")
cm4 = confusion_matrix(Y_test, knn.predict(X_test))
print(cm4)
print(accuracy_score(Y_test, knn.predict(X_test)))




# comparision table
models = {
    "Logistic Regression": (lreg, y_pred_lreg),
    "Random Forest": (rf, y_pred_rf),
    "Decision Tree": (dt, y_pred_dt),
    "K-Nearest Neighbors": (knn, y_pred_knn)
}

for name, (model, y_pred) in models.items():
    print(f"\n{name}")
    print("Confusion Matrix:")
    print(confusion_matrix(Y_test, y_pred))
    print("Accuracy:", accuracy_score(Y_test, y_pred))
    print("Classification Report:")
    print(classification_report(Y_test, y_pred, target_names=["Non-diabetic", "Diabetic"]))
    
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(Y_test, y_prob)
        print("ROC AUC Score:", roc_auc)
    except:
        print("ROC AUC Score: Not available (no probability prediction support)")



# saving model and scaler
joblib.dump(lreg, 'models/logistic_regression.pkl')
joblib.dump(rf, 'models/random_forest_classifier.pkl')
joblib.dump(dt, 'models/decision_tree_classifier.pkl')
joblib.dump(knn, 'models/k_nearest_neighbors.pkl')
joblib.dump(sc, 'scaler/scaler.pkl')

print("Models and scaler saved successfully!")