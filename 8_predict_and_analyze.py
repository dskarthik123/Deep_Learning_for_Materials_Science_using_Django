import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
def load_data(file_path):
    """Load the dataset from a CSV file."""
    data = pd.read_csv(file_path)
    print(f"Data loaded successfully from: {file_path}")
    return data

def main():
    # Define file paths and target column
    data_file_path = 'D:/#Python Programs/Final Project/Data/processed_data_with_new_features.csv'
    model_file_path = 'best_random_forest_model.pkl'
    target_column = 'Energy Above Hull'  # Update this to your actual target column name

    # Load data
    data = load_data(data_file_path)

    # Check columns in the DataFrame
    print("Columns available in the DataFrame:", data.columns.tolist())

    # Prepare features and target variable
    if target_column not in data.columns:
        raise KeyError(f"Target column '{target_column}' not found in the DataFrame")

    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # Load the saved model
    model = joblib.load(model_file_path)
    print(f"Model loaded successfully from: {model_file_path}")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate model performance
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

    # Analyze results
    # Plot actual vs predicted values
    plt.figure(figsize=(7, 4))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', lw=2)
    plt.grid(True)
    plt.savefig('D:/#Python Programs/Final Project/actual_vs_predicted.png')
    plt.show()

    # Plot residuals
    residuals = y_test - y_pred
    plt.figure(figsize=(7, 4))
    sns.histplot(residuals, bins=30, kde=True)
    plt.xlabel('Residuals')
    plt.title('Distribution of Residuals')
    plt.grid(True)
    plt.savefig('D:/#Python Programs/Final Project/distribution_of_residuals.png')
    plt.show()

if __name__ == "__main__":
    main()