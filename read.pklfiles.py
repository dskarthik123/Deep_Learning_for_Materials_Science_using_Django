import joblib

# Load the model
model = joblib.load('best_random_forest_model.pkl')

# Check if the model has attribute 'feature_importances_'
if hasattr(model, 'feature_importances_'):
    print("Feature Importances:")
    print(model.feature_importances_)

# Check if the model has a method to get feature names (like in some cases)
if hasattr(model, 'feature_names_in_'):
    print("Feature Names:")
    print(model.feature_names_in_)
