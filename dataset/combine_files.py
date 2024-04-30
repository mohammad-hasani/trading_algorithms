import os
import pandas as pd

# Path to the folder containing CSV files
folder_path = './ETC/'

# Get a list of all CSV files in the folder
csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

csv_files = sorted(csv_files)

# Initialize an empty DataFrame to store the combined data
combined_data = pd.DataFrame()

# Loop through each CSV file and append its data to the combined DataFrame
for file in csv_files:
    file_path = os.path.join(folder_path, file)
    data = pd.read_csv(file_path)
    combined_data = combined_data._append(data, ignore_index=True)

# Save the combined data to a new CSV file
combined_data.to_csv('./BTC/ETCUSDT.csv', index=False)

print("Combination complete. Check 'combined_data.csv' in the specified output folder.")
