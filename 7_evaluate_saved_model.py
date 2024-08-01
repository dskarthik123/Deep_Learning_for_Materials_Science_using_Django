import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load data
def load_data(file_path):
    """Load the dataset from a CSV file."""
    data = pd.read_csv(file_path)
    print(f"Data loaded successfully from: {file_path}")
    return data

def main():
    # Define file paths
    data_file_path = 'D:/#Python Programs/Final Project/Data/processed_data_with_new_features.csv'
    model_file_path = 'best_random_forest_model.pkl'

    # Load data
    data = load_data(data_file_path)
    
    # Define target column
    target_column = 'Energy Above Hull'  # Update this to your actual target column name

    # Check if the target column exists in the DataFrame
    if target_column not in data.columns:
        raise KeyError(f"Target column '{target_column}' not found in the DataFrame")

    # Prepare features and target variable
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load the saved model
    model = joblib.load(model_file_path)
    print(f"Model type: {type(model)}")  # Check the type of the loaded model
    
    if not hasattr(model, 'predict'):
        raise TypeError("The loaded model does not have a 'predict' method. It may not be a valid sklearn model.")

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluate model performance
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

if __name__ == "__main__":
    main()