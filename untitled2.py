import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load dataset
file_path = r"C:\Users\Mohammed Sham\Downloads\boston_house_prices.csv"

# Your CSV has a junk header, so use header=1
df = pd.read_csv(file_path, header=1)

# Clean dataset
df.columns = ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT","MEDV"]
df = df[df["MEDV"].notna()]  # if last row has NaN values

# Splitting
X = df.drop("MEDV", axis=1)
y = df["MEDV"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================
# 1️⃣ RANDOMIZED SEARCH CV
# ==============================
rf = RandomForestRegressor()

random_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, 30, 40],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["auto", "sqrt"]
}

random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=random_grid,
    n_iter=10,
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

print("Running RandomizedSearchCV...")
random_search.fit(X_train_scaled, y_train)
print("Best Params (RandomSearch):", random_search.best_params_)

# ==============================
# 2️⃣ GRID SEARCH CV
# ==============================
param_grid = {
    "n_estimators": [random_search.best_params_['n_estimators']],
    "max_depth": [random_search.best_params_['max_depth']]
}

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,
    verbose=1,
    n_jobs=-1
)

print("Running GridSearchCV...")
grid_search.fit(X_train_scaled, y_train)
print("Best Params (GridSearch):", grid_search.best_params_)

# Best model
best_model = grid_search.best_estimator_

# ==============================
# Evaluation
# ==============================
y_pred = best_model.predict(X_test_scaled)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n========== MODEL PERFORMANCE ==========")
print("RMSE:", rmse)
print("R2 Score:", r2)

# Save model & scaler
joblib.dump(best_model, "boston_house_model.joblib")
joblib.dump(scaler, "scaler.joblib")

print("\nModel and scaler saved successfully!")
