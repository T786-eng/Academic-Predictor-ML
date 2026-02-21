import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def load_data(file_path):
    """Loads the dataset and returns features and target."""
    print(f"📥 Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    X = df[['Attendance_Hours']]
    y = df['Final_Marks']
    return X, y

def train_model(X, y):
    """Splits data and trains a Linear Regression model."""
    # Splitting into 80% Training and 20% Testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

def evaluate_and_plot(model, X, y, X_test, y_test):
    """Calculates metrics and generates the visualization."""
    y_pred = model.predict(X_test)
    
    # Metrics calculation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\n📊 Model Performance:")
    print(f" - Mean Squared Error (MSE): {mse:.4f}")
    print(f" - R² Score (Accuracy): {r2:.4f}")
    print(f" - Coefficient (Slope): {model.coef_[0]:.4f}")
    print(f" - Intercept: {model.intercept_:.4f}")

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='#3498db', alpha=0.5, label='Actual Data Points')
    plt.plot(X_test, y_pred, color='#e74c3c', linewidth=2.5, label='Regression Trendline')
    
    plt.title('Student Performance Analysis: Attendance vs. Marks', fontsize=14)
    plt.xlabel('Attendance Hours', fontsize=12)
    plt.ylabel('Final Marks', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Saving the result
    plt.tight_layout()
    plt.savefig('analysis_result.png', dpi=300)
    print("\n✅ Visualization saved as 'analysis_result.png'")

if __name__ == "__main__":
    FILE_NAME = 'Study_vs_Score_data.csv'
    
    try:
        X, y = load_data(FILE_NAME)
        model, X_test, y_test = train_model(X, y)
        evaluate_and_plot(model, X, y, X_test, y_test)
    except Exception as e:
        print(f"❌ An error occurred: {e}")