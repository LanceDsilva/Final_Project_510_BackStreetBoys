"""
Script: Renewables_data_processing.py
Purpose:
    - Processes renewable energy stock data for model training and prediction.
    - Calculates features like moving averages, daily returns, and volatility.

Inputs:
    - renewable_stock_data.csv
Outputs:
    - Preprocessed data saved as preprocessed_renewable_stock_data.csv.
"""


# Import libraries
import pandas as pd
import numpy as np

# Load the datasets (replace with actual file paths)
renewable_stock_data = pd.read_csv('renewable_stock_data.csv', parse_dates=['Date'])
renewable_sector_data = pd.read_csv('renewable_sector_data.csv', parse_dates=['Date'])

# Define timezone-aware date range
# start_date = pd.Timestamp('2012-03-30', tz='UTC')
# end_date = pd.Timestamp('2024-12-31', tz='UTC')

# Convert 'Date' column to datetime and remove timezone awareness
renewable_stock_data['Date'] = pd.to_datetime(renewable_stock_data['Date'], utc=True).dt.tz_convert(None)
renewable_sector_data['Date'] = pd.to_datetime(renewable_sector_data['Date'], utc=True).dt.tz_convert(None)

# Define date range without timezone awareness
start_date = pd.Timestamp('2012-01-01')
end_date = pd.Timestamp('2024-12-31')

# Filter the data
renewable_stock_data = renewable_stock_data[(renewable_stock_data['Date'] >= start_date) & 
                                            (renewable_stock_data['Date'] <= end_date)]
renewable_sector_data = renewable_sector_data[(renewable_sector_data['Date'] >= start_date) & 
                                              (renewable_sector_data['Date'] <= end_date)]

# Display the filtered data
print("Filtered Oil and Gas Stock Data:")
print(renewable_stock_data.head())

print("\nFiltered Oil and Gas Sector Data:")
print(renewable_sector_data.head())

# Check for missing values in oil and gas stock data
print("Missing Values in Oil and Gas Stock Data:")
print(renewable_stock_data.isnull().sum())

# Check for missing values in sector data
print("\nMissing Values in Oil and Gas Sector Data:")
print(renewable_sector_data.isnull().sum())

# Calculate daily returns for individual stocks and sector
renewable_stock_data['Daily_Return'] = renewable_stock_data.groupby('Ticker')['Close'].pct_change()
renewable_sector_data['Daily_Return'] = renewable_sector_data['Close'].pct_change()

# Drop rows with NaN in Daily_Return (optional)
renewable_stock_data = renewable_stock_data.dropna(subset=['Daily_Return'])
renewable_sector_data = renewable_sector_data.dropna(subset=['Daily_Return'])

# Display the cleaned data
print("Cleaned Oil and Gas Stock Data with Daily Returns:")
print(renewable_stock_data[['Date', 'Ticker', 'Close', 'Daily_Return']].head())

print("\nCleaned Oil and Gas Sector Data with Daily Returns:")
print(renewable_sector_data[['Date', 'Close', 'Daily_Return']].head())

# Add moving averages (7-day and 30-day) for trends
for window in [7, 30]:  # Weekly and monthly moving averages
    renewable_stock_data[f'MA_{window}'] = renewable_stock_data.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window).mean())
    renewable_sector_data[f'MA_{window}'] = renewable_sector_data['Close'].rolling(window).mean()

# Display a preview of the moving averages
print("Moving Averages in Oil and Gas Stock Data:")
print(renewable_stock_data[['Date', 'Ticker', 'Close', 'MA_7', 'MA_30']].tail())

print("\nMoving Averages in Sector Data:")
print(renewable_sector_data[['Date', 'Close', 'MA_7', 'MA_30']].tail())


# Calculate volatility (30-day rolling standard deviation of daily returns)
renewable_stock_data['Volatility'] = renewable_stock_data.groupby('Ticker')['Daily_Return'].transform(lambda x: x.rolling(30).std())
renewable_sector_data['Volatility'] = renewable_sector_data['Daily_Return'].rolling(30).std()

# Display a preview of the volatility
print("Volatility in Oil and Gas Stock Data:")
print(renewable_stock_data[['Date', 'Ticker', 'Daily_Return', 'Volatility']].tail())

print("\nVolatility in Sector Data:")
print(renewable_sector_data[['Date', 'Daily_Return', 'Volatility']].tail())

# Normalize the Volume column using Log Transformation
renewable_stock_data['Log_Volume'] = np.log1p(renewable_stock_data['Volume'])
renewable_sector_data['Log_Volume'] = np.log1p(renewable_sector_data['Volume'])

# Display a preview of the normalized volume
print("Normalized Volume in Oil and Gas Stock Data:")
print(renewable_stock_data[['Date', 'Ticker', 'Volume', 'Log_Volume']].head())

print("\nNormalized Volume in Sector Data:")
print(renewable_sector_data[['Date', 'Volume', 'Log_Volume']].head())

# Drop unnecessary columns
renewable_stock_data.drop(columns=['Dividends', 'Stock Splits'], inplace=True)
renewable_sector_data.drop(columns=['Dividends', 'Stock Splits', 'Capital Gains'], inplace=True, errors='ignore')

# Display the final structure of the preprocessed data
print("Final Structure of Oil and Gas Stock Data:")
print(renewable_stock_data.head())

print("\nFinal Structure of Sector Data:")
print(renewable_sector_data.head())

# Save the preprocessed data to CSV files for future use
renewable_stock_data.to_csv('preprocessed_renewable_stock_data.csv', index=False)
renewable_sector_data.to_csv('preprocessed_renewable_sector_data.csv', index=False)

print("Preprocessed data has been saved to CSV files.")
