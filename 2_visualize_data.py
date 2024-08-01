import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def load_data(file_path):
    """Load the dataset from a CSV file."""
    data = pd.read_csv(file_path)
    print(f"Data loaded successfully from: {file_path}")
    return data

def plot_histograms(data, numeric_columns, save_path):
    """Plot histograms for numeric columns."""
    for col in numeric_columns:
        plt.figure(figsize=(7, 4))
        sns.histplot(data[col].dropna(), kde=True, bins=30)
        plt.title(f'Histogram of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'{col}_histogram.png'))  # Save the histogram plot as an image file
        plt.show()

def plot_boxplots(data, numeric_columns, save_path):
    """Plot boxplots for numeric columns."""
    for col in numeric_columns:
        plt.figure(figsize=(7, 4))
        sns.boxplot(y=data[col].dropna())  # Use `y` instead of `x` for boxplot
        plt.title(f'Boxplot of {col}')
        plt.xlabel(col)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'{col}_boxplot.png'))  # Save the boxplot as an image file
        plt.show()

def plot_correlation_heatmap(data, numeric_columns, save_path):
    """Plot a heatmap of the correlation matrix."""
    plt.figure(figsize=(7, 4))
    correlation_matrix = data[numeric_columns].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'correlation_heatmap.png'))  # Save the heatmap as an image file
    plt.show()

if __name__ == "__main__":
    file_path = 'D:/#Python Programs/Final Project/Data/processed_data.csv'
    save_path = 'D:/#Python Programs/Final Project/Visualized charts'
    
    # Create the save path directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    data = load_data(file_path)
    
    numeric_columns = [
        'Energy Above Hull', 'Formation Energy', 'Volume', 'Density', 
        'Total Magnetization', 'Bulk Modulus, Voigt', 'Bulk Modulus, Reuss', 
        'Bulk Modulus, VRH', 'Shear Modulus, Voigt', 'Shear Modulus, Reuss', 
        'Shear Modulus, VRH', 'Elastic Anisotropy', 'Weighted Surface Energy', 
        'Surface Anisotropy', 'Shape Factor', 'Work Function', 
        'Piezoelectric Modulus', 'Total Dielectric Constant', 
        'Ionic Dielectric Constant', 'Static Dielectric Constant'
    ]

    plot_histograms(data, numeric_columns, save_path)
    plot_boxplots(data, numeric_columns, save_path)
    plot_correlation_heatmap(data, numeric_columns, save_path)