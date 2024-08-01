import pandas as pd
import joblib

# Load the model
model = joblib.load('best_random_forest_model.pkl')

# Load the test data
data = pd.read_csv('D:/#Python Programs/Final Project/Data/processed_data_with_new_features.csv')  # Replace with your actual test data file

# Define the feature engineering function used during training
def feature_engineering(df):
    # Apply feature engineering steps used during training
    # Modify these lines based on actual feature engineering steps
    df['Energy Above Hull Total Magnetization'] = df['Energy Above Hull.1'] * df['Total Magnetization.1']
    return df

# Apply the same feature engineering to the test data
data = feature_engineering(data)

# Ensure columns are in the expected order and check for missing columns
expected_columns = [
    'Density', 'Total Magnetization', 'Is Metal_False', 'Is Metal_True',
    'Density_binned', 'Energy Above Hull.1', 'Density.1', 'Total Magnetization.1',
    'Energy Above Hull^2', 'Energy Above Hull Density',
    'Energy Above Hull Total Magnetization', 'Density^2',
    'Density Total Magnetization', 'Total Magnetization^2',
    'PCA1', 'PCA2', 'PCA3', 'PCA4'
]

# Check for missing columns
missing_columns = [col for col in expected_columns if col not in data.columns]
if missing_columns:
    raise ValueError(f"Missing columns: {', '.join(missing_columns)}")

# Reorder columns to match expected order
data = data[expected_columns]

# Apply the same scaling as used during training (if applicable)
# Note: Replace 'scaler.pkl' with the correct scaler path if you have one
# scaler = joblib.load('scaler.pkl')
# X_test_scaled = scaler.transform(data)

# Here we are just preparing data without scaling
X_test = data[expected_columns]

# Make prediction
predictions = model.predict(X_test)
print(f"Predictions: {predictions}")