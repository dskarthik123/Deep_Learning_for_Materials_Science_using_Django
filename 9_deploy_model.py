from flask import Flask, request, jsonify
import joblib
import pandas as pd
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Retrieve the model path from environment variables
model_path = os.getenv('MODEL_PATH')

# Debugging statements
print("Environment variables loaded")
print(f"FLASK_ENV: {os.getenv('FLASK_ENV')}")
print(f"MODEL_PATH: {model_path}")

# Check if model_path is set
if not model_path:
    raise ValueError("MODEL_PATH environment variable not set. Check your .env file.")

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
try:
    model = joblib.load(model_path)
except Exception as e:
    raise ValueError(f"Error loading model: {str(e)}")

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the POST request
        data = request.get_json(force=True)
        
        # Convert the data into a DataFrame
        df = pd.DataFrame([data])  # Use [data] to ensure it's a single row DataFrame
        
        # Define the expected columns in the correct order
        expected_columns = [
            'Density', 'Total Magnetization', 'Is Metal_False', 'Is Metal_True',
            'Density_binned', 'Energy Above Hull.1', 'Density.1', 'Total Magnetization.1',
            'Energy Above Hull^2', 'Energy Above Hull Density',
            'Energy Above Hull Total Magnetization', 'Density^2',
            'Density Total Magnetization', 'Total Magnetization^2',
            'PCA1', 'PCA2', 'PCA3', 'PCA4'
        ]

        # Check if the data has all required columns and order
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            return jsonify({'error': f'Missing columns: {", ".join(missing_columns)}'}), 400
        
        # Ensure columns are in the expected order
        df = df[expected_columns]

        # Make prediction
        predictions = model.predict(df)
        
        # Return the predictions as JSON
        return jsonify({'prediction': predictions.tolist()})
    
    except Exception as e:
        # Return the error message as JSON
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)