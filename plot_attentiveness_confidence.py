import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned CSV
df = pd.read_csv(r"C:\Users\sahil\Desktop\attentiveness_detection\attentiveness_log_cleaned.csv")

# Convert 'Time' to datetime (so it can be plotted on the x-axis)
df['Time'] = pd.to_datetime(df['Time'])

# Plot Confidence over Time for each Class
plt.figure(figsize=(10, 6))  # Set the size of the plot
sns.lineplot(data=df, x='Time', y='Confidence', hue='Class', marker='o')  # Plot the data
plt.title('Confidence Over Time for Different Attentiveness Classes')  # Add a title
plt.xlabel('Time')  # Label for the x-axis
plt.ylabel('Confidence')  # Label for the y-axis
plt.xticks(rotation=45)  # Rotate the x-axis labels for better visibility
plt.tight_layout()  # Ensure everything fits nicely in the plot
plt.show()  # Display the plot
