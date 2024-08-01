import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def load_data(file_path):
    """Load the dataset from a CSV file."""
    data = pd.read_csv(file_path)
    print(f"Data loaded successfully from: {file_path}")
    return data

def preprocess_data(data):
    """Preprocess the dataset."""
    
    # Convert non-numeric columns to numeric, coercing errors to NaN
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Define numeric columns and target variable
    target_column = 'Band Gap'
    numeric_columns = [
        'Energy Above Hull', 'Formation Energy', 'Volume', 'Density', 
        'Total Magnetization', 'Bulk Modulus, Voigt', 'Bulk Modulus, Reuss', 
        'Bulk Modulus, VRH', 'Shear Modulus, Voigt', 'Shear Modulus, Reuss', 
        'Shear Modulus, VRH', 'Elastic Anisotropy', 'Weighted Surface Energy', 
        'Surface Anisotropy', 'Shape Factor', 'Work Function', 'Piezoelectric Modulus', 
        'Total Dielectric Constant', 'Ionic Dielectric Constant', 'Static Dielectric Constant'
    ]
    categorical_columns = [
        'Crystal System', 'Space Group Symbol', 'Magnetic Ordering', 
        'Is Gap Direct', 'Is Metal'
    ]
    
    # Print columns being used
    print("Numeric columns:", numeric_columns)
    print("Categorical columns:", categorical_columns)
    
    # Handle infinite values by replacing them with NaN
    data.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
    
    # Identify columns with all missing values
    all_missing_columns = [col for col in numeric_columns if data[col].isnull().all()]
    if all_missing_columns:
        print("Columns with all missing values:", all_missing_columns)
        numeric_columns = [col for col in numeric_columns if col not in all_missing_columns]
    
    # Fill missing values for numeric columns
    imputer = SimpleImputer(strategy='mean')
    if numeric_columns:
        data[numeric_columns] = imputer.fit_transform(data[numeric_columns])
    
    # Convert boolean columns to strings
    for col in categorical_columns:
        if data[col].dtype == 'bool':
            data[col] = data[col].astype(str)
    
    # Handle categorical columns with all missing values
    non_empty_categorical_columns = [col for col in categorical_columns if data[col].notna().any()]
    if non_empty_categorical_columns:
        # Fill missing values for non-empty categorical columns
        imputer_cat = SimpleImputer(strategy='most_frequent')
        data[non_empty_categorical_columns] = imputer_cat.fit_transform(data[non_empty_categorical_columns])
    else:
        print("No categorical columns with non-missing values to process.")
    
    # Check for remaining missing or infinite values
    if data.isnull().sum().any():
        print("Remaining missing values:")
        print(data.isnull().sum())
    
    # Normalize numeric features
    if numeric_columns:
        scaler = StandardScaler()
        data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
    
    # One-hot encoding categorical features
    if non_empty_categorical_columns:
        data = pd.get_dummies(data, columns=non_empty_categorical_columns)
    
    # Ensure the target column is retained
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' is not in the dataset.")
    
    return data

if __name__ == "__main__":
    file_path = 'D:/#Python Programs/Final Project/Data/Materials_data.csv'
    data = load_data(file_path)
    processed_data = preprocess_data(data)
    processed_data.to_csv('D:/#Python Programs/Final Project/Data/processed_data.csv', index=False)
    print("Processed data saved to: processed_data.csv")