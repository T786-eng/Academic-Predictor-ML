# Study Hours vs. Final Score Analysis

This project uses Simple Linear Regression to analyze and predict the relationship between a student's attendance/study hours and their final marks. Using a dataset of 1,000 records, the model identifies how strongly attendance correlates with academic performance.

## 📋 Project Overview

The core of this project is main.py, which performs the following tasks:

- **Data Loading**: Reads the Study_vs_Score_data.csv file.
- **Data Splitting**: Divides the data into Training (80%) and Testing (20%) sets to ensure the model generalizes well to new data.
- **Linear Regression**: Fits a mathematical line (y=mx+b) to the data.
- **Evaluation**: Calculates the accuracy of the model using Mean Squared Error (MSE) and the R-squared (R²) score.
- **Visualization**: Generates a scatter plot with the regression line overlaid to visualize the trend.

## 🚀 Getting Started

### Prerequisites

You will need Python installed along with the following libraries:

- pandas (for data manipulation)
- matplotlib (for plotting)
- scikit-learn (for the machine learning model)

You can install these via pip:

```bash
pip install pandas matplotlib scikit-learn
```

### Installation & Usage

1. Ensure main.py and Study_vs_Score_data.csv are in the same directory.
2. Run the script:

```bash
python main.py
```

## 📊 Understanding the Output

### Model Metrics

When you run the script, it will print performance metrics to the console:

- **R-squared Score (~0.82)**: This indicates that approximately 82% of the variation in final marks can be explained by attendance hours. A value closer to 1.0 indicates a very strong relationship.
- **Coefficient**: This represents the "slope." It tells you how many marks a student is expected to gain for every additional hour of attendance.

### Visualization

The script generates a file named analysis_result.png.

- **Blue Dots**: Represent the actual data points from your CSV.
- **Red Line**: Represents the model's prediction. The closer the blue dots are to this red line, the more accurate the model is.

## 📂 File Structure

- main.py: The Python script containing the logic.
- Study_vs_Score_data.csv: The dataset containing Attendance_Hours and Final_Marks.
- analysis_result.png: The visual output generated after running the script.
- README.md: Project documentation.
