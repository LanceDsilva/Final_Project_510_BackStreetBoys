# Project: Trump - Bump or Slump? Predicting Stock Market Valuations for Oil/Gas and Renewable Energy Sectors

## Team Members
- **Ryan Norring**  
- **Lance Dsilva**

---

## Overview

This project analyzes and predicts the valuations of oil/gas and renewable energy sectors, focusing on the impact of political administrations and other factors. Predictions are generated for 2025–2028 using historical trends and machine learning.

---

## Problem Statement

We examine how political administrations and economic conditions influence stock valuations in the oil/gas and renewable sectors. By leveraging historical data and machine learning, we predict future stock performance.

---

## How to Run the Project

Follow these steps to run the project:

### Step 1: Install Python and Conda
1. Ensure Python is installed on your system.
   - Download and install Python from [python.org](https://python.org/downloads).
2. Install Conda (optional but recommended).
   - Download and install Conda from [Anaconda](https://www.anaconda.com/products/distribution).

---

### Step 2: Clone the GitHub Repository
1. Open a terminal or command prompt.
2. Clone the repository from GitHub:
   ```bash
   git clone <repository-link>
   cd <repository-name>

### Step 3: Set Up the Environment

1. **Using Conda**:
   - Create a new Conda environment:
     ```bash
     conda create --name stock-env python=3.9 -y
     ```
   - Activate the environment:
     ```bash
     conda activate stock-env
     ```

2. **Install Required Libraries**:
   - Install all the dependencies listed in the `requirements.txt` file:
     ```bash
     pip install -r requirements.txt
     ```

### Step 4: Run the Project

1. **Execute the Main Script**:
   - Run the main script to execute the full workflow:
     ```bash
     python main.py
     ```
   - This script will:
     - Fetch and process data.
     - Train the Ridge Regression model.
     - Generate predictions for 2025–2028.
     - Produce exploratory data analysis (EDA) outputs.

---

## File Structure

- **`data_collection.py`**:
  Fetches stock and macroeconomic data and saves it to CSV files.

- **`oil_and_gas_data_preprocessing.py`**:
  Preprocesses oil/gas stock data, calculates features, and saves processed data.

- **`Renewables_data_processing.py`**:
  Similar preprocessing for renewable energy stocks.

- **`model_training.py`**:
  Trains a Ridge Regression model using the processed data and saves the trained model.

- **`predict.py`**:
  Uses the trained model to predict stock prices for 2025–2028 and generates plots.

- **`EDA.py`**:
  Performs exploratory data analysis (EDA), including trend plots and correlation heatmaps.

- **`main.py`**:
  Executes the entire workflow.

---


## Dependencies

- Python 3.7+
- Libraries listed in `requirements.txt`.

---

## Notes

- Ensure an internet connection to fetch data via APIs.
- Replace `XOM` and `FSLR` with your preferred stock tickers, if necessary.

---

Let me know if there are further edits or clarifications needed!
