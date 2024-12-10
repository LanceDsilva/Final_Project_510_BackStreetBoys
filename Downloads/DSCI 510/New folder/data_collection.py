"""
Script: data_collection.py
Purpose: Collect historical stock data for oil/gas and renewable energy sectors using Yahoo Finance.
Inputs: None (fetches live data using `yfinance`).
Outputs:
    - oil_gas_stock_data.csv
    - oil_gas_sector_data.csv
    - renewable_stock_data.csv
    - renewable_sector_data.csv
"""

import yfinance as yf
import pandas as pd

# Oil and Gas Sector and stock data
# Define the stock tickers and sector index
oil_gas_stocks = ['XOM', 'CVX', 'BP']  # Oil and gas stock tickers
sector_index = 'XLE'  # Example index for oil & gas

# Set the date range for data collection
start_date = '2012-01-01'
end_date = '2024-12-31'


# Function to fetch stock or index data with a daily interval
def fetch_stock_data(ticker, start, end):
    """
    Fetch stock data from Yahoo Finance for a given ticker and date range.

    Parameters:
        ticker (str): Stock ticker symbol (e.g., 'XOM').
        start_date (str): Start date for data retrieval (format: 'YYYY-MM-DD').
        end_date (str): End date for data retrieval (format: 'YYYY-MM-DD').

    Returns:
        pd.DataFrame: Historical stock data with an added 'Ticker' column.
    """
    stock = yf.Ticker(ticker)
    # Fetch data with a valid interval
    data = stock.history(start=start, end=end, interval="1d")  # Specify daily interval
    data['Ticker'] = ticker  # Add a column for the ticker
    return data


# Collect data for individual stocks
stock_data_list = []
for ticker in oil_gas_stocks:
    print(f"Fetching data for {ticker}...")
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    stock_data_list.append(stock_data)

# Combine all stock data into one DataFrame
oil_gas_stock_data = pd.concat(stock_data_list)
oil_gas_stock_data.reset_index(inplace=True)

# Collect data for the oil and gas sector index
print(f"Fetching sector index data for {sector_index}...")
sector_data = fetch_stock_data(sector_index, start_date, end_date)
sector_data.reset_index(inplace=True)

# Preview the collected stock data
print("\nOil and Gas Stock Data Preview:")
print(oil_gas_stock_data.head())

# Preview the collected sector index data
print("\nOil and Gas Sector Data Preview:")
print(sector_data.head())

# Save the data to CSV files
oil_gas_stock_data.to_csv('oil_gas_stock_data.csv', index=False)
sector_data.to_csv('oil_gas_sector_data.csv', index=False)

print("\nData collection completed and saved to CSV files.")


# Renewables Sector and stock data
# Define the stock tickers and sector index
renewables_stocks = ['FSLR', 'NEE', 'ENPH']  # Renewables stock tickers
renew_sector_index = 'ICLN'  # Example index for oil & gas

# Set the date range for data collection
renew_start_date = '2012-03-30' # most recent IPO date (ENPH)
renew_end_date = '2024-12-31'


# Function to fetch stock or index data with a daily interval
def fetch_stock_data(ticker, start, end):
    stock = yf.Ticker(ticker)
    # Fetch data with a valid interval
    data = stock.history(start=start, end=end, interval="1d")  # Specify daily interval
    data['Ticker'] = ticker  # Add a column for the ticker
    return data


# Collect data for individual stocks
renewable_stock_data_list = []
for ticker in renewables_stocks:
    print(f"Fetching data for {ticker}...")
    stock_data = fetch_stock_data(ticker, renew_start_date, renew_end_date)
    renewable_stock_data_list.append(stock_data)

# Combine all stock data into one DataFrame
renewable_stock_data = pd.concat(renewable_stock_data_list)
renewable_stock_data.reset_index(inplace=True)

# Collect data for the renewables sector index
print(f"Fetching sector index data for {renew_sector_index}...")
renewable_sector_data = fetch_stock_data(renew_sector_index, renew_start_date, renew_end_date)
renewable_sector_data.reset_index(inplace=True)

# Preview the collected stock data
print("\nRenewables Stock Data Preview:")
print(renewable_stock_data.head())

# Preview the collected sector index data
print("\nRenewables Sector Data Preview:")
print(renewable_sector_data.head())

# Save the data to CSV files
renewable_stock_data.to_csv('renewable_stock_data.csv', index=False)
renewable_sector_data.to_csv('renewable_sector_data.csv', index=False)

print("\nData collection completed and saved to CSV files.")
