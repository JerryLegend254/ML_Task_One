# Nairobi Office Price Prediction Model

## Background
A simple linear regression model to predict office prices in Nairobi using gradient descent optimization.

## Dataset Description
- 14 records
- Features: location, amenities, size, price
- Used for training a linear regression model

## Problem Description
- Linear regression model to predict office prices
- One feature (x): office size
- Target variable (y): office price
- Goal: Learn parameters m and c for equation y = mx + c

## Implementation
### Model Components
- Mean Squared Error (MSE) as Performance Measure
- Gradient Descent as Learning Algorithm

### Formulas Used
- Linear Equation: y = mx + c
- MSE = (1/n) * Σ(y_actual - y_predicted)²
- Gradient Descent Updates:
  * m = m - learning_rate * d(MSE)/dm 
  * c = c - learning_rate * d(MSE)/dc

### Parameters
- Learning rate: 0.01
- Epochs: 10
- Momentum: 0.9

## Results
- Final MSE: - [223.2070]
- Slope (m): [0.4637]
- Intercept (c): [0.0000]
- Predicted price for 100 sq ft: $[108.73]

## Requirements
- Python 3.x
- NumPy
- Pandas
- Matplotlib
