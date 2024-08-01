import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib

# Load the data
data = pd.read_csv('D:/#Python Programs/Final Project/Data/processed_data_with_new_features.csv')

# # Remove duplicate columns
# data = data.loc[:, ~data.columns.duplicated()]

# Print available columns
print("Columns available in the DataFrame:", data.columns.tolist())

# Define the target column based on the actual column name
target_column = 'Energy Above Hull'  # Replace with the correct column name based on your data

# Check if the target column exists in the DataFrame
if target_column not in data.columns:
    raise KeyError(f"Target column '{target_column}' not found in the DataFrame")

# Define features and target
X = data.drop(target_column, axis=1)
y = data[target_column]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = RandomForestRegressor(random_state=42)

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2', None]  # Ensure valid values only
}

# Set up Grid Search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                           cv=3, scoring='neg_mean_squared_error', 
                           verbose=1, n_jobs=-1)

# Perform Grid Search
grid_search.fit(X_train, y_train)

# Get the best parameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Evaluate the best model
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Save the best model
joblib.dump(best_model, 'best_random_forest_model.pkl')

# Print results
print("Best Parameters:", best_params)
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)
