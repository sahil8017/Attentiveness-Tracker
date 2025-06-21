import pandas as pd
import os

# Path to the CSV file
csv_path = r"C:\Users\sahil\Desktop\attentiveness_detection\attentiveness_log.csv"

# Check if the file exists
if os.path.exists(csv_path):
    # Load the CSV (assuming it's comma-separated)
    try:
        df = pd.read_csv(csv_path, sep=",")  # Comma-separated
    except:
        print("Failed to load CSV. Please check the delimiter in the file.")
        exit(1)

    # Print the first few rows to inspect the data
    print("First few rows of the CSV data:")
    print(df.head())

    # Check the number of columns
    print(f"Number of columns in the data: {df.shape[1]}")

    # Clean and format the columns (Only proceed if the data has 4 columns)
    if df.shape[1] == 4:
        df.columns = ['Time', 'Class', 'Confidence', 'Frame_ID']  # Ensure the correct columns

        # Remove any leading/trailing spaces from the column names
        df.columns = df.columns.str.strip()

        # Check if all required columns are present
        if {"Time", "Class", "Confidence", "Frame_ID"}.issubset(df.columns):
            print("CSV is correctly formatted!")
        else:
            print("CSV is missing some required columns.")
            print("The required columns are: Time, Class, Confidence, Frame_ID")
            exit(1)

        # Check the first few rows to ensure the format is correct
        print("Preview of cleaned CSV data:")
        print(df.head())

        # Save the cleaned version as a new CSV file
        cleaned_csv_path = r"C:\Users\sahil\Desktop\attentiveness_detection\attentiveness_log_cleaned.csv"
        df.to_csv(cleaned_csv_path, index=False)
        print(f"Cleaned CSV saved to: {cleaned_csv_path}")
    else:
        print("CSV does not have the expected number of columns. Please check the format.")
else:
    print(f"File not found: {csv_path}")
