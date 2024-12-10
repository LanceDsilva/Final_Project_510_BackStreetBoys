"""
Script: model_training.py
Purpose:
    - Trains a Ridge Regression model using preprocessed datasets for oil/gas and renewable energy sectors.
    - Performs cross-validation and residual analysis to evaluate model performance.

Inputs:
    - Preprocessed data files (e.g., preprocessed_oil_gas_stock_data.csv).
Outputs:
    - Trained Ridge Regression model (ridge_model.pkl).
    - Console outputs for model performance metrics.
"""


# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import joblib


# Step 1: Load Data
renewable_stock_data = pd.read_csv("renewable_stock_data.csv")
oil_gas_stock_data = pd.read_csv("oil_gas_stock_data.csv")
combined_stock_data = pd.concat([renewable_stock_data, oil_gas_stock_data], ignore_index=True)

# Step 2: Generate Missing Features
if 'Daily_Return' not in combined_stock_data.columns:
    combined_stock_data['Daily_Return'] = combined_stock_data.groupby('Ticker')['Close'].pct_change()

if 'MA_30' not in combined_stock_data.columns:
    combined_stock_data['MA_30'] = combined_stock_data.groupby('Ticker')['Close'].transform(lambda x: x.rolling(30).mean())

if 'Volatility' not in combined_stock_data.columns:
    combined_stock_data['Volatility'] = combined_stock_data.groupby('Ticker')['Close'].transform(lambda x: x.pct_change().rolling(30).std())

if 'Log_Volume' not in combined_stock_data.columns:
    combined_stock_data['Log_Volume'] = np.log1p(combined_stock_data['Volume'])

# Step 3: Define Features and Target
required_features = ['Daily_Return', 'MA_30', 'Volatility', 'Log_Volume']
X = combined_stock_data[required_features].dropna()
y = combined_stock_data['Close'].iloc[X.index]

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train Ridge Regression Model
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)

# Step 6: Evaluate the Model
y_pred = ridge_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# Perform 5-fold cross-validation
cv_scores = cross_val_score(ridge_model, X, y, cv=5, scoring='r2')

# Print cross-validation results
print("Cross-Validation R² Scores:", cv_scores)
print("Mean Cross-Validation R²:", cv_scores.mean())

# Residual analysis
residuals = y_test - y_pred

# Plot residuals
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--', label='Zero Error Line')
plt.xlabel('Predicted Prices')
plt.ylabel('Residuals')
plt.title('Residual Analysis: Predicted vs Residuals')
plt.legend()
plt.grid()
plt.show()


# Test different alpha values
alphas = [0.1, 1, 10, 100]
for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    y_pred_alpha = ridge.predict(X_test)
    mse_alpha = mean_squared_error(y_test, y_pred_alpha)
    r2_alpha = r2_score(y_test, y_pred_alpha)
    print(f"Alpha: {alpha}, MSE: {mse_alpha:.2f}, R²: {r2_alpha:.2f}")

# Save the Ridge model
joblib.dump(ridge_model, 'ridge_model.pkl')
print("Model saved successfully!")


# Ensure the predictions align with actual test data
aligned_predictions = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
}, index=X_test.index)

# Display the actual vs predicted table
print("Actual vs Predicted Table:")
print(aligned_predictions)
# Look for any renaming or dropping operations
print(renewable_stock_data.head())
print(oil_gas_stock_data.head())