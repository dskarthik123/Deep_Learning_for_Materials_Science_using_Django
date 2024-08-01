import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Load data
def load_data(file_path):
    """Load the dataset from a CSV file."""
    data = pd.read_csv(file_path)
    print(f"Data loaded successfully from: {file_path}")
    return data

def main():
    # Define file path and target column
    file_path = 'D:/#Python Programs/Final Project/Data/processed_data_with_new_features.csv'
    target_column = 'Energy Above Hull'  # Update this to your actual target column name

    # Load data
    data = load_data(file_path)
    
    # Check columns in the DataFrame
    print("Columns available in the DataFrame:", data.columns.tolist())
    
    # Prepare features and target variable
    if target_column not in data.columns:
        raise KeyError(f"Target column '{target_column}' not found in the DataFrame")

    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a pipeline with a RandomForestRegressor
    model = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=100, random_state=42))
    
    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluate model performance
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

    # Perform cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"Cross-Validation R^2 Scores: {cv_scores}")
    print(f"Mean Cross-Validation R^2 Score: {cv_scores.mean()}")

if __name__ == "__main__":
    main()