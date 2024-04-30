import os
import pandas as pd

# Replace 'your_directory' with the actual path to your directory containing CSV files
directory_path = './ETC/'

# Define the headers
headers = [
    'open_time',
    'open',
    'high',
    'low',
    'close',
    'volume',
    'close_time',
    'quote_volume',
    'count',
    'taker_buy_volume',
    'taker_buy_quote_volume',
    'ignore'
]

# Iterate through all files in the directory
for filename in os.listdir(directory_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(directory_path, filename)

        # Read the CSV file without header
        df = pd.read_csv(file_path, header=None)

        # Set the headers
        df.columns = headers

        # Save the updated DataFrame back to the CSV file
        df.to_csv(file_path, index=False)

        print(f"Headers set for {filename}")

# You can add additional processing or print statements as needed
