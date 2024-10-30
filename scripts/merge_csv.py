import os
import pandas as pd

# Define the base directory where all folders are located
base_dir = '/Users/orenw/Documents/tb_skills_analysis/data/SDF_UserStudy_Data/extracted_metrics'

# Initialize an empty list to store dataframes
all_data = []

# Iterate over each folder in the base directory
for folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder)
    
    # Check if the path is a directory (folder)
    if os.path.isdir(folder_path):
        # Define the file path for the CSV file in the folder
        csv_file = os.path.join(folder_path, 'timed_stroke_metrics.csv')
        
        # Check if the CSV file exists in the folder
        if os.path.exists(csv_file):
            # Read the CSV file
            df = pd.read_csv(csv_file)
            
            # Ensure the dataframe is not empty and there's more than 1 row of data
            if not df.empty and len(df.dropna(how='all', subset=df.columns.difference(['Source']))) > 1:
                # Add a 'source' column with the folder name
                df['Source'] = folder
                
                # Drop rows where all values except the 'source' are NaN
                df = df.dropna(how='all', subset=df.columns.difference(['Source']))
                
                # Append the dataframe to the list
                all_data.append(df)

# Concatenate all dataframes into one
combined_df = pd.concat(all_data, ignore_index=True)

# Save the combined dataframe to a new CSV file if needed
combined_df.to_csv('/Users/orenw/Documents/tb_skills_analysis/data/SDF_UserStudy_Data/extracted_metrics/combined_extracted_stroke_metrics.csv', index=False)