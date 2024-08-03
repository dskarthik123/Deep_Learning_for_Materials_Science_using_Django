import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

# Load the model
model = joblib.load('D:/#Python Programs/Final Project/best_random_forest_model.pkl')

# Load the dataset
data = pd.read_csv('D:/#Python Programs/Final Project/Data/processed_data_with_new_features.csv')  # Replace with your actual test data file

# Prepare the data
features = [
    'Density', 'Total Magnetization', 'Is Metal_False', 'Is Metal_True',
    'Density_binned', 'Energy Above Hull.1', 'Density.1', 'Total Magnetization.1',
    'Energy Above Hull^2', 'Energy Above Hull Density',
    'Energy Above Hull Total Magnetization', 'Density^2',
    'Density Total Magnetization', 'Total Magnetization^2',
    'PCA1', 'PCA2', 'PCA3', 'PCA4'
]

X = data[features]
y_true = data['Energy Above Hull']  # Replace 'target_column_name' with your actual target column name

# Make predictions
y_pred = model.predict(X)

# Calculate metrics
mse = mean_squared_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")