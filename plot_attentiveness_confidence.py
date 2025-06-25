import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os # Import os for file existence check

# Path to the cleaned CSV file
csv_path = r"C:\Users\sahil\Desktop\attentiveness_detection\attentiveness_log_cleaned.csv"

# Check if the file exists
if not os.path.exists(csv_path):
    print(f"Error: File not found at '{csv_path}'")
    print("Please ensure the attentiveness tracking Streamlit app has been run and generated log data,")
    print("and that the cleaning script has successfully created 'attentiveness_log_cleaned.csv'.")
    exit(1) # Exit if file doesn't exist

# Load the cleaned CSV
try:
    df = pd.read_csv(csv_path) # Pandas is usually smart enough to detect comma separator
except pd.errors.EmptyDataError:
    print(f"Error: The file '{csv_path}' is empty or contains no data to parse.")
    print("Please ensure the attentiveness tracking Streamlit app has recorded data.")
    exit(1)
except Exception as e:
    print(f"An unexpected error occurred while loading the CSV: {e}")
    exit(1)

# Check if the DataFrame is empty after loading (e.g., only header, no rows)
if df.empty:
    print(f"Warning: The CSV file '{csv_path}' was loaded, but it contains no data rows.")
    print("No plot will be generated. Please ensure the tracking app recorded data.")
    exit(0) # Exit gracefully if no data rows

# Convert 'Time' to datetime
df['Time'] = pd.to_datetime(df['Time'])

# Ensure required columns exist, especially if the cleaning step wasn't run
required_columns = {"Time", "Class", "Confidence", "Frame_ID"}
if not required_columns.issubset(df.columns):
    print(f"Error: Missing required columns in the CSV. Found: {df.columns.tolist()}")
    print(f"Expected: {list(required_columns)}")
    print("Please ensure the data is correctly formatted, potentially by running the cleaning script first.")
    exit(1)

# Plot Confidence over Time for each Class
plt.figure(figsize=(12, 7)) # Slightly larger figure for better detail
sns.lineplot(data=df, x='Time', y='Confidence', hue='Class', marker='o', linewidth=2.5) # Thicker lines

plt.title('Confidence Over Time for Different Attentiveness Classes', fontsize=16)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Confidence', fontsize=12)
plt.xticks(rotation=45, ha='right') # Align rotated labels nicely
plt.yticks(np.arange(0, 1.1, 0.1)) # Set y-axis ticks from 0 to 1 with 0.1 increments
plt.grid(True, linestyle='--', alpha=0.7) # Add a subtle grid
plt.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left') # Move legend outside to prevent overlap
plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout to make space for the external legend

plt.show()

