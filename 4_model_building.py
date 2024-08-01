import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, r2_score

# Load the processed data
data = pd.read_csv('D:/#Python Programs/Final Project/Data/processed_data_with_new_features.csv')

# Print columns to find the correct target column name
print("Columns available in the DataFrame:", data.columns.tolist())

# Define features (X) and target (y)
# Replace 'YourTargetColumnName' with the actual target column name
target_column = 'Energy Above Hull'  # Example: replace with your actual target column name
X = data.drop(target_column, axis=1)
y = data[target_column]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train models
def train_classification_model():
    # Initialize the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    print("Random Forest Classifier")
    print("Accuracy Score:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

def train_regression_model():
    # Initialize the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    print("Random Forest Regressor")
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    print("R^2 Score:", r2_score(y_test, y_pred))

if __name__ == "__main__":
    # Choose the appropriate model based on your problem type
    problem_type = 'regression'  # Use 'classification' for classification problems
    
    if problem_type == 'classification':
        train_classification_model()
    elif problem_type == 'regression':
        train_regression_model()
    else:
        print("Invalid problem type specified. Please set 'classification' or 'regression'.")
