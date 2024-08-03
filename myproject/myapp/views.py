from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import joblib
import pandas as pd
from dotenv import load_dotenv
import os
import json
from django.http import HttpResponse
# Load environment variables from .env file
load_dotenv()

# Retrieve the model path from environment variables
model_path = os.getenv('MODEL_PATH')

# Load the trained model
model = joblib.load(model_path)

@csrf_exempt
def predict(request):
    if request.method == 'POST':
        try:
            # Parse JSON data from the request body
            data = json.loads(request.body)

            # Convert the data into a DataFrame
            df = pd.DataFrame([data])

            # Define the expected columns in the correct order
            expected_columns = [
                'Density', 'Total Magnetization', 'Is Metal_False', 'Is Metal_True',
                'Density_binned', 'Energy Above Hull.1', 'Density.1', 'Total Magnetization.1',
                'Energy Above Hull^2', 'Energy Above Hull Density',
                'Energy Above Hull Total Magnetization', 'Density^2',
                'Density Total Magnetization', 'Total Magnetization^2',
                'PCA1', 'PCA2', 'PCA3', 'PCA4'
            ]

            # Check for missing columns
            missing_columns = [col for col in expected_columns if col not in df.columns]
            if missing_columns:
                return JsonResponse({'error': f'Missing columns: {", ".join(missing_columns)}'}, status=400)

            # Ensure columns are in the expected order
            df = df[expected_columns]

            # Make prediction
            predictions = model.predict(df)

            # Return the predictions as JSON
            return JsonResponse({'prediction': predictions.tolist()})

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid HTTP method'}, status=405)

def home(request):
    return HttpResponse("Welcome to the home page!")