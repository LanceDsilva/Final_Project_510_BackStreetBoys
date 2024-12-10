"""
Script: EDA.py
Purpose:
    - Performs exploratory data analysis on preprocessed data.
    - Generates visualizations like trend plots and heatmaps.
    - Summarizes sector performance across periods and administrations.

Inputs:
    - oil_gas_stock_data.csv
    - renewable_stock_data.csv
Outputs:
    - Trend plots for closing prices.
    - Correlation heatmaps.
    - Console summaries of sector performance.
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import renewable stock data
renewable_stock_data = pd.read_csv('renewable_stock_data.csv', parse_dates=['Date'])
renewable_stock_data['Date'] = pd.to_datetime(renewable_stock_data['Date'], utc=True).dt.tz_localize(None)

# Import oil and gas stock data
oil_gas_stock_data = pd.read_csv('oil_gas_stock_data.csv', parse_dates=['Date'])
oil_gas_stock_data['Date'] = pd.to_datetime(oil_gas_stock_data['Date'], utc=True).dt.tz_localize(None)

# Plot renewable stock closing prices over time
plt.figure(figsize=(12, 6))
for ticker, group in renewable_stock_data.groupby('Ticker'):
    plt.plot(group['Date'], group['Close'], label=ticker)

plt.title("Closing Prices of Renewable Stock Tickers Over Time")
plt.xlabel("Time")
plt.ylabel("Close Price")
plt.legend(title="Tickers")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot oil & gas stock closing prices over time
plt.figure(figsize=(12, 6))
for ticker, group in oil_gas_stock_data.groupby('Ticker'):
    plt.plot(group['Date'], group['Close'], label=ticker)

plt.title("Closing Prices of Oil & Gas Stock Tickers Over Time")
plt.xlabel("Time")
plt.ylabel("Close Price")
plt.legend(title="Tickers")
plt.grid(True)
plt.tight_layout()
plt.show()

# Ensuring data is sorted by Ticker and Date
renewable_stock_data.sort_values(by=['Ticker', 'Date'], inplace=True)
oil_gas_stock_data.sort_values(by=['Ticker', 'Date'], inplace=True)

# Defining periods of interest
periods = {
    '2012-2015': ('2012-01-01', '2015-12-31'),
    '2016-2019': ('2016-01-01', '2019-12-31'),
    '2020-2024': ('2020-01-01', '2024-12-31')
}

# Analyze renewable stocks by period
renewable_results = []
for period_name, (start_date, end_date) in periods.items():
    period_data = renewable_stock_data[
        (renewable_stock_data['Date'] >= start_date) & (renewable_stock_data['Date'] <= end_date)
    ]
    metrics = period_data.groupby('Ticker').apply(
        lambda group: pd.Series({
            'First_Close': group['Close'].iloc[0],
            'Last_Close': group['Close'].iloc[-1],
            'Pct_Change': ((group['Close'].iloc[-1] - group['Close'].iloc[0]) / group['Close'].iloc[0]) * 100,
            'Volatility_Score': group['Close'].pct_change().std() * 100
        })
    ).reset_index()
    metrics['Period'] = period_name
    renewable_results.append(metrics)

# Combine renewable results
final_results_renewable = pd.concat(renewable_results, ignore_index=True)
print("Summary of Close Prices and Percentage Changes for Renewable Stocks by Period:")
print(final_results_renewable)

# Analyze oil & gas stocks by period
oil_gas_results = []
for period_name, (start_date, end_date) in periods.items():
    period_data = oil_gas_stock_data[
        (oil_gas_stock_data['Date'] >= start_date) & (oil_gas_stock_data['Date'] <= end_date)
    ]
    metrics = period_data.groupby('Ticker').apply(
        lambda group: pd.Series({
            'First_Close': group['Close'].iloc[0],
            'Last_Close': group['Close'].iloc[-1],
            'Pct_Change': ((group['Close'].iloc[-1] - group['Close'].iloc[0]) / group['Close'].iloc[0]) * 100,
            'Volatility_Score': group['Close'].pct_change().std() * 100
        })
    ).reset_index()
    metrics['Period'] = period_name
    oil_gas_results.append(metrics)

# Combine oil & gas results
final_results_oil_gas = pd.concat(oil_gas_results, ignore_index=True)
print("Summary of Close Prices and Percentage Changes for Oil & Gas Stocks by Period:")
print(final_results_oil_gas)

# Correlation heatmap for renewable stocks
pivot_close_renewables = renewable_stock_data.pivot(index='Date', columns='Ticker', values='Close')
correlation_matrix_renewables = pivot_close_renewables.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix_renewables, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Matrix of Renewable Stock Tickers")
plt.show()

# Correlation heatmap for oil & gas stocks
pivot_close_oil_gas = oil_gas_stock_data.pivot(index='Date', columns='Ticker', values='Close')
correlation_matrix_oil_gas = pivot_close_oil_gas.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix_oil_gas, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Matrix of Oil & Gas Stock Tickers")
plt.show()

# Combined analysis for all stocks
combined_data = pd.concat([renewable_stock_data, oil_gas_stock_data])
combined_data.sort_values(by=['Date', 'Ticker'], inplace=True)
pivot_close_combined = combined_data.pivot(index='Date', columns='Ticker', values='Close')
correlation_matrix_combined = pivot_close_combined.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_combined, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Matrix of All Stock Tickers (Renewables + Oil/Gas)")
plt.show()
