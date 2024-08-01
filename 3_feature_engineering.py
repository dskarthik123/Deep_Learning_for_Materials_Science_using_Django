import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.preprocessing import KBinsDiscretizer

def load_data(file_path):
    """Load the dataset from a CSV file."""
    data = pd.read_csv(file_path)
    print(f"Data loaded successfully from: {file_path}")
    return data

def feature_selection(data):
    """Perform feature selection using methods like RFE or feature importance."""
    # Print columns in the DataFrame to verify
    print("Columns available in the DataFrame:", data.columns.tolist())
    
    # Example feature names (Update these according to your actual features)
    selected_features = ['Energy Above Hull', 'Density', 'Total Magnetization', 'Is Metal_False', 'Is Metal_True']
    
    # Ensure the selected features are in the DataFrame columns
    valid_features = [feature for feature in selected_features if feature in data.columns]
    print(f"Valid selected features: {valid_features}")

    # Return the data with selected features
    if valid_features:
        return data[valid_features]
    else:
        print("No valid features found in the DataFrame.")
        return pd.DataFrame()  # Return an empty DataFrame or handle this case as needed

def feature_creation(data):
    """Create new features from the dataset."""
    
    # Check if the data is empty
    if data.empty:
        print("No data available for feature creation.")
        return data
    
    # Impute missing values before creating polynomial features
    imputer = SimpleImputer(strategy='mean')
    
    # Ensure only numeric data is passed for imputation and polynomial features
    numeric_columns = data.select_dtypes(include=['number']).columns
    data_imputed = pd.DataFrame(imputer.fit_transform(data[numeric_columns]), columns=numeric_columns, index=data.index)
    
    # Create polynomial features (Ensure this applies to numeric data only)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(data_imputed)
    poly_feature_names = poly.get_feature_names_out(numeric_columns)
    data_poly = pd.DataFrame(X_poly, columns=poly_feature_names, index=data.index)
    
    # Perform PCA for dimensionality reduction
    pca = PCA(n_components=0.95)  # Adjust the explained variance ratio as needed
    data_pca = pd.DataFrame(pca.fit_transform(data_poly), columns=[f'PCA{i+1}' for i in range(pca.n_components_)])
    
    # Binning for certain features (example: binning numeric columns)
    binning_features = ['Density', 'Volume']  # Example features to bin
    binning = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
    for feature in binning_features:
        if feature in data.columns:
            data_binned = pd.DataFrame(binning.fit_transform(data[[feature]]), columns=[f'{feature}_binned'])
            data = pd.concat([data, data_binned], axis=1)
    
    # Concatenate all new features
    data_with_new_features = pd.concat([data, data_poly, data_pca], axis=1)
    
    # Save the processed data with new features
    data_with_new_features.to_csv('D:/#Python Programs/Final Project/Data/processed_data_with_new_features.csv', index=False)
    print("Data with new features saved to: processed_data_with_new_features.csv")
    
    return data_with_new_features

if __name__ == "__main__":
    file_path = 'D:/#Python Programs/Final Project/Data/processed_data.csv'
    data = load_data(file_path)
    
    # Perform feature selection
    data_selected = feature_selection(data)
    
    # Create new features
    data_with_new_features = feature_creation(data_selected)