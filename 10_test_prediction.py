import joblib
import pandas as pd

# Load the model
model = joblib.load('best_random_forest_model.pkl')

# Sample input data
sample_data = {
    "Density": 1.2,
    "Total Magnetization": 0.3,
    "Is Metal_False": 1,
    "Is Metal_True": 0,
    "Density_binned": 1,
    "Energy Above Hull.1": 0.01,
    "Density.1": 1.44,
    "Total Magnetization.1": 0.36,
    "Energy Above Hull^2": 0.01,
    "Energy Above Hull Density": 0.12,
    "Energy Above Hull Total Magnetization": 0.03,
    "Density^2": 1.44,
    "Density Total Magnetization": 0.36,
    "Total Magnetization^2": 0.09,
    "PCA1": 0.5,
    "PCA2": 0.3,
    "PCA3": 0.2,
    "PCA4": 0.1
}

# Convert to DataFrame
df = pd.DataFrame([sample_data])

# Define the expected columns
expected_columns = [
    'Density', 'Total Magnetization', 'Is Metal_False', 'Is Metal_True',
    'Density_binned', 'Energy Above Hull.1', 'Density.1', 'Total Magnetization.1',
    'Energy Above Hull^2', 'Energy Above Hull Density',
    'Energy Above Hull Total Magnetization', 'Density^2',
    'Density Total Magnetization', 'Total Magnetization^2',
    'PCA1', 'PCA2', 'PCA3', 'PCA4'
]

# Add missing columns with default values if necessary
for col in expected_columns:
    if col not in df.columns:
        df[col] = 0

# Reorder columns to match expected order
df = df[expected_columns]

# Make prediction
prediction = model.predict(df)
print(f"Local prediction: {prediction[0]}")