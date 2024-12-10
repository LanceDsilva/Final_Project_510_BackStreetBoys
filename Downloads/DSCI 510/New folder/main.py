"""
Script: main.py
Purpose:
    - Coordinates the execution of all scripts in the project.
    - Ensures correct order of execution for data collection, preprocessing, training, predictions, and EDA.

Inputs: None.
Outputs: Executes all scripts and generates outputs specified in each script.
"""

import subprocess


def run_script(script_name):
    """
    Executes a Python script using subprocess and handles errors.

    Parameters:
        script_name (str): Name of the script to run (e.g., 'data_collection.py').

    Returns:
        None
    """
    try:
        print(f"Running {script_name}...")
        subprocess.run(["python", script_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script_name}: {e}")


def main():
    # Running the Python scripts in order
    run_script("data_collection.py")
    run_script("oil_and_gas_data_preprocessing.py")
    run_script("Renewables_data_processing.py")
    run_script("model_training.py")
    run_script("predict.py")
    run_script("EDA.py")


if __name__ == "__main__":
    main()
