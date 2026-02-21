"""
Study Hours vs. Final Score Analysis
====================================

This module performs Simple Linear Regression analysis to predict the relationship 
between a student's attendance hours and their final marks. It loads data, trains a 
model, evaluates its performance, and generates a visualization.

Author: Data Analysis
Date: 2026
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os
import numpy as np


def load_and_prepare_data(file_path):
    """
    Loads the dataset and prepares features and target variables.
    
    Args:
        file_path (str): Path to the CSV file containing the dataset.
        
    Returns:
        tuple: (X, y) where X is the feature matrix (Attendance_Hours) 
               and y is the target vector (Final_Marks).
               
    Raises:
        FileNotFoundError: If the CSV file does not exist.
        KeyError: If required columns are missing from the dataset.
    """
    print("Loading data from:", file_path)
    data = pd.read_csv(file_path)
    
    # Validate that required columns exist
    if 'Attendance_Hours' not in data.columns or 'Final_Marks' not in data.columns:
        raise KeyError("Dataset must contain 'Attendance_Hours' and 'Final_Marks' columns.")
    
    X = data[['Attendance_Hours']]
    y = data['Final_Marks']
    
    print(f"✓ Data loaded successfully! Records: {len(data)}")
    return X, y


def train_and_evaluate(X, y):
    """
    Splits data, trains a Linear Regression model, and evaluates its performance.
    
    Args:
        X (pd.DataFrame): Feature matrix (independent variable).
        y (pd.Series): Target vector (dependent variable).
        
    Returns:
        tuple: (model, X_test, y_test, y_pred, metrics) containing:
            - model: Trained LinearRegression object
            - X_test: Test feature set
            - y_test: Test target values
            - y_pred: Model predictions on test set
            - metrics: Dictionary with MSE and R² scores
    """
    print("\n" + "="*50)
    print("Training Linear Regression Model...")
    print("="*50)
    
    # Split into 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training Set: {len(X_train)} samples | Test Set: {len(X_test)} samples")
    
    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate Metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Display Model Performance
    print("\n" + "-"*50)
    print("Model Performance Metrics")
    print("-"*50)
    print(f"Mean Squared Error (MSE):     {mse:.2f}")
    print(f"R² Score:                     {r2:.4f}")
    print(f"Model Equation: y = {model.coef_[0]:.4f}x + {model.intercept_:.4f}")
    print("-"*50)
    
    metrics = {'mse': mse, 'r2': r2}
    return model, X_test, y_test, y_pred, metrics


def plot_results(X, y, X_test, y_test, y_pred, metrics, output_dir):
    """
    Generates and saves a professional scatter plot with the regression line.
    
    Args:
        X (pd.DataFrame): Complete feature set (for plotting all data points).
        y (pd.Series): Complete target set (for plotting all data points).
        X_test (pd.DataFrame): Test feature set.
        y_test (pd.Series): Test target values.
        y_pred (np.array): Model predictions.
        metrics (dict): Dictionary containing 'mse' and 'r2' values.
        output_dir (str): Directory to save the plot image.
    """
    print("\nGenerating visualization...")
    
    # Create figure with high DPI for better quality
    fig, ax = plt.subplots(figsize=(12, 7), dpi=100)
    
    # Plot actual data points
    ax.scatter(X, y, color='#3498db', alpha=0.5, s=50, 
               label='Actual Data Points', edgecolors='none')
    
    # Sort X_test for proper line plotting
    sort_idx = np.argsort(X_test.values.flatten())
    X_test_sorted = X_test.values[sort_idx]
    y_pred_sorted = y_pred[sort_idx]
    
    # Plot regression line
    ax.plot(X_test_sorted, y_pred_sorted, color='#e74c3c', linewidth=3, 
            label='Regression Line (Prediction)', zorder=5)
    
    # Customize the plot
    ax.set_title('Impact of Attendance on Final Marks', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Attendance Hours', fontsize=12, fontweight='bold')
    ax.set_ylabel('Final Marks', fontsize=12, fontweight='bold')
    
    # Add subtitle with R² value
    r2_value = metrics['r2']
    subtitle = f'R² Score: {r2_value:.4f} (Explains {r2_value*100:.2f}% of variance)'
    ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, 
            ha='center', fontsize=11, style='italic', color='#555555')
    
    # Enhance grid
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Add legend
    ax.legend(fontsize=10, loc='upper left', framealpha=0.95)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, 'analysis_result.png')
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"✓ Visualization saved as '{output_path}'")
    
    # Show the plot
    plt.show()



def main():
    """
    Main execution function that orchestrates the entire analysis pipeline.
    """
    print("="*60)
    print("  Study Hours vs. Final Score - Linear Regression Analysis")
    print("="*60)
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = os.path.join(script_dir, 'Study_vs_Score_data.csv')
    
    try:
        # Step 1: Load Data
        print("\nStep 1: Data Loading")
        X, y = load_and_prepare_data(file_name)
        
        # Step 2: Train and Evaluate
        print("\nStep 2: Model Training & Evaluation")
        model, X_test, y_test, y_pred, metrics = train_and_evaluate(X, y)
        
        # Step 3: Visualize
        print("\nStep 3: Visualization")
        plot_results(X, y, X_test, y_test, y_pred, metrics, script_dir)
        
        print("\n" + "="*60)
        print("✓ Analysis completed successfully!")
        print("="*60)
        
    except FileNotFoundError:
        print(f"\n❌ Error: The file '{file_name}' was not found.")
        print(f"   Please ensure 'Study_vs_Score_data.csv' is in the same directory as this script.")
        
    except KeyError as e:
        print(f"\n❌ Error: Missing required column in dataset.")
        print(f"   {str(e)}")
        print(f"   The dataset must contain 'Attendance_Hours' and 'Final_Marks' columns.")
        
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        print(f"   Please check your data and try again.")


if __name__ == '__main__':
    main()