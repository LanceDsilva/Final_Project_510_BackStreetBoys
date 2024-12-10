"""
Script: predict.py
Purpose:
    - Uses the trained Ridge Regression model to predict stock prices for 2025–2028.
    - Generates predictions for both oil/gas and renewable energy sectors.
    - Outputs predictions as CSV files and visualizations.

Inputs:
    - ridge_model.pkl (trained Ridge Regression model).
    - Preprocessed datasets for oil/gas and renewable stocks.
Outputs:
    - Predicted stock prices (CSV and plots).
"""


import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load('ridge_model.pkl')

# Define future years
future_years = [2025, 2026, 2027, 2028]


# Function to ensure all required features are present
def ensure_features(data, ticker):
    """
    Ensure the presence of essential features in the stock data for a specific ticker.

    This function filters the dataset for the specified stock ticker and checks if the required features 
    ('Daily_Return', 'MA_30', 'Volatility', 'Log_Volume') are present. If any feature is missing, it calculates 
    and adds them to the dataset. Finally, it removes rows with missing values generated during feature calculations.

    Parameters:
        data (pd.DataFrame): The input stock dataset containing columns such as 'Ticker', 'Close', and 'Volume'.
        ticker (str): The stock ticker symbol for which the features need to be ensured (e.g., 'XOM').

    Returns:
        pd.DataFrame: The filtered and updated dataset with the required features and no missing values. 
    Features Calculated:
        - Daily_Return: Percentage change in closing price for the stock.
        - MA_30: 30-day moving average of the closing price.
        - Volatility: 30-day rolling standard deviation of daily returns.
        - Log_Volume: Natural logarithm of the stock's traded volume.
    """
    # Filter data for the ticker
    data = data[data['Ticker'] == ticker]

    # Check and calculate missing features
    if 'Daily_Return' not in data.columns:
        data['Daily_Return'] = data.groupby('Ticker')['Close'].pct_change()
    if 'MA_30' not in data.columns:
        data['MA_30'] = data.groupby('Ticker')['Close'].transform(lambda x: x.rolling(30).mean())
    if 'Volatility' not in data.columns:
        data['Volatility'] = data.groupby('Ticker')['Daily_Return'].transform(lambda x: x.rolling(30).std())
    if 'Log_Volume' not in data.columns:
        data['Log_Volume'] = np.log1p(data['Volume'])

    # Drop rows with NaN values generated during calculations
    data = data.dropna(subset=['Daily_Return', 'MA_30', 'Volatility', 'Log_Volume'])

    return data


# Function to predict for a specific stock or index
def predict_future_prices(ticker, preprocessed_data_file, output_csv, plot_title, plot_file):
    """
    Predict future stock prices for a specific stock or sector index.

    Parameters:
        ticker (str): Stock ticker symbol (e.g., 'XOM').
        preprocessed_data_file (str): Path to the preprocessed data file.
        output_csv (str): Path to save the prediction results as a CSV file.
        plot_title (str): Title for the prediction plot.
        plot_file (str): Path to save the prediction plot.

    Returns:
        pd.DataFrame: DataFrame containing the predictions for the specified years.
    """
    # Load preprocessed data
    preprocessed_data = pd.read_csv(preprocessed_data_file)

    # Ensure required features are present
    preprocessed_data = ensure_features(preprocessed_data, ticker)

    # Filter data for the specified ticker
    stock_data = preprocessed_data[preprocessed_data['Ticker'] == ticker]

    # Check if stock_data is empty
    if stock_data.empty:
        print(f"Error: No data found for ticker {ticker} in {preprocessed_data_file}.")
        return None

    # Extract the latest values of features to estimate future features
    recent_data = stock_data.tail(30)  # Use the last 30 days of data
    latest_ma_30 = recent_data['MA_30'].mean()
    latest_volatility = recent_data['Volatility'].mean()
    latest_daily_return = recent_data['Daily_Return'].mean()
    latest_log_volume = recent_data['Log_Volume'].mean()

    # Estimate future features based on recent trends
    import random

    # Simulate variability in feature projections
    future_features = {
        'Daily_Return': [latest_daily_return * (1 + random.uniform(-0.01, 0.02)) for i in range(4)],
        'MA_30': [latest_ma_30 + random.uniform(5, 15) * i for i in range(4)],  # Randomized increments
        'Volatility': [latest_volatility * (1 + random.uniform(-0.01, 0.02)) for i in range(4)],
        'Log_Volume': [latest_log_volume + random.uniform(0.1, 0.5) * i for i in range(4)]  # Simulate stronger volume growth
    }

    # Create a DataFrame for the predictions
    future_data = pd.DataFrame(future_features, index=future_years)

    # Predict future prices
    predicted_prices = model.predict(future_data)

    # Prepare results in a tabular format
    predictions_table = pd.DataFrame({
        'Year': future_years,
        'Predicted_Price': predicted_prices
    })

    # Save the predictions to a CSV file
    predictions_table.to_csv(output_csv, index=False)

    # Print the table to the console
    print(f"\nPredicted Prices for {ticker} (2025-2028):")
    print(predictions_table)

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(predictions_table['Year'], predictions_table['Predicted_Price'], marker='o', linestyle='-', linewidth=2)
    plt.title(plot_title, fontsize=16)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Predicted Price', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(predictions_table['Year'])
    plt.tight_layout()
    plt.savefig(plot_file)  # Save the plot
    plt.show()

    return predictions_table


# Predictions for Oil and Gas Stock
oil_gas_stock_predictions = predict_future_prices(
    ticker='XOM',
    preprocessed_data_file='preprocessed_oil_gas_stock_data.csv',
    output_csv='future_predictions_XOM.csv',
    plot_title='Predicted Prices for XOM (2025-2028)',
    plot_file='future_predictions_XOM_plot.png'
)

# Predictions for Renewable Stock
renewable_stock_predictions = predict_future_prices(
    ticker='FSLR',
    preprocessed_data_file='preprocessed_renewable_stock_data.csv',
    output_csv='future_predictions_FSLR.csv',
    plot_title='Predicted Prices for FSLR (2025-2028)',
    plot_file='future_predictions_FSLR_plot.png'
)

# Predictions for Oil and Gas Sector Index
oil_gas_sector_predictions = predict_future_prices(
    ticker='XLE',
    preprocessed_data_file='preprocessed_sector_data.csv',
    output_csv='future_predictions_XLE.csv',
    plot_title='Predicted Prices for Oil and Gas Sector (XLE) (2025-2028)',
    plot_file='future_predictions_XLE_plot.png'
)

# Predictions for Renewable Sector Index
renewable_sector_predictions = predict_future_prices(
    ticker='ICLN',
    preprocessed_data_file='preprocessed_renewable_sector_data.csv',
    output_csv='future_predictions_ICLN.csv',
    plot_title='Predicted Prices for Renewable Sector (ICLN) (2025-2028)',
    plot_file='future_predictions_ICLN_plot.png'
)

print("All predictions completed and saved to CSV and plots!")
