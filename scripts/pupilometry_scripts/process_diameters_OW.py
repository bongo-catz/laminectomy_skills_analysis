import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


csv_file_path = "/Users/orenw/Documents/tb_skills_analysis/data/SDF_UserStudy_Data/exp_dirs_DONOTOVERWRITE.csv"
df_csv = pd.read_csv(csv_file_path)

"""
# Load CSV with all trial folders
csv_file_path = "/Users/orenw/Documents/tb_skills_analysis/data/SDF_UserStudy_Data/exp_dirs_DONOTOVERWRITE.csv"
experiments = pd.read_csv(csv_file_path)['exp_dir'].tolist()
raw_diameters = [exp + "/000/extracted_diameter.csv" for exp in experiments]
"""

# Initialize an empty DataFrame to collect results
df_sum = pd.DataFrame()

for index, row in df_csv.iterrows():
    exp = row['exp_dir']
    # Build the file path
    file = os.path.join(exp, "000", "extracted_diameter.csv")
    try:
        df = pd.read_csv(file)

        # Get the value of the timestamp in the fifth row
        fifth_row_timestamp = df['timestamp'].iloc[4]

        # Check if any of the first four rows have a timestamp that differs from the fifth row by more than 1000
        rows_to_delete = df['timestamp'].iloc[:4].apply(lambda x: abs(x - fifth_row_timestamp) > 1000)

        # Delete those rows
        df = df.drop(df.index[:4][rows_to_delete])

        # Ensure the 'topic' column is treated as strings and handle NaN values
        df['topic'] = df['topic'].astype(str)

        # Filter out rows where the topic contains "2d"
        df = df[~df['topic'].str.contains('3d')]

        # Remove rows in which both diameters are 0
        df = df[~((df['diameter_2d [px]'] == 0) & (df['diameter_3d [mm]'] == 0))]
        df = df[~((df['diameter_2d [px]'] == 0))]

        # Treat the first timestamp as t=0 seconds and calculate time in seconds
        start_time = df['timestamp'].min()
        df['time_seconds'] = df['timestamp'] - start_time

        # Filter data for the right and left eyes
        right_eye_data = df[df['eye_id'] == 0.0]['diameter_2d [px]']
        left_eye_data = df[df['eye_id'] == 1.0]['diameter_2d [px]']

        # Calculate average and standard deviation for both eyes
        right_eye_mean = right_eye_data.mean()
        right_eye_std = right_eye_data.std()
        left_eye_mean = left_eye_data.mean()
        left_eye_std = left_eye_data.std()
        both_eyes_mean = df['diameter_2d [px]'].mean()
        both_eyes_std = df['diameter_2d [px]'].std()

        # Create a DataFrame with one row for the current results
        df_row = pd.DataFrame({
            'filepath': [file],
            'diameter_right_eye_mean': [right_eye_mean],
            'diameter_right_eye_std': [right_eye_std],
            'diameter_left_eye_mean': [left_eye_mean],
            'diameter_left_eye_std': [left_eye_std],
            'diameter_both_eyes_mean': [both_eyes_mean],
            'diameter_both_eyes_std': [both_eyes_std]
        })

        # Include other columns and values from the corresponding row in csv_file_path
        df_row = pd.concat([df_row.reset_index(drop=True), row.to_frame().T.reset_index(drop=True)], axis=1)

        # Append the new row to df_sum
        df_sum = pd.concat([df_sum, df_row], ignore_index=True)

        print(f"Processed file: {file}")

    except FileNotFoundError:
        print(f"File not found: {file}, skipping...")
        continue
    except pd.errors.EmptyDataError:
        print(f"Empty data in file: {file}, skipping...")
        continue

# Save df_sum to a new master Excel sheet
df_head = df_sum.head(10)
print(df_head)
master_csv_file = "/Users/orenw/Documents/tb_skills_analysis/data/SDF_UserStudy_Data/2d diameter_TLX_summary.csv"
df_sum.to_csv(master_csv_file, index=False)


