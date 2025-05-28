import os
import numpy as np
import pandas as pd
from collections import defaultdict
from classes.exp_reader import ExpReader
from pathlib import Path
from classes.plotting_skills_analysis import StrokeMetricsVisualizer, plot_3d_vx_rmvd
from classes.stroke_metrics import StrokeMetrics
from classes.stroke_extractor import StrokeExtractor
from classes.gen_metrics import GenMetrics
import h5py
import click
import logging as log

# example usage: python3 scripts/metric_extractor.py --exp-csv data/Laminectomy_Data/exp_dirs.csv --output-base-dir output_laminectomy
@click.command()
@click.option('--exp-csv', required=True, type=click.Path(exists=True),
              help='Path to the CSV file containing experiment directories.')
@click.option('--output-base-dir', required=True, type=click.Path(),
              help='Path to base directory for saving output metrics.')
@click.option("--log-path", required=True, type=click.Path(),
              help='Path to log file to see outputs.')
def main(exp_csv, output_base_dir, log_path):
    exp_csv = pd.read_csv(exp_csv)
    exps = exp_csv['exp_dir']
    
    log.basicConfig(filename=log_path, filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    level=log.INFO)
    log.info(f"Logging to file: {log_path}")

    for i in range(len(exps)):
        exp_dir = exps[i]
        log.info(f"Processing: {exp_dir}")

        # Check if the directory exists
        if not os.path.exists(exp_dir):
            log.info(f"Directory does not exist: {exp_dir}. Skipping.")
            continue  # Skip to the next iteration if the directory doesn't exist

        # Check if there are any files in the experiment directory
        num_files = len([f for f in os.listdir(exp_dir) if os.path.isfile(os.path.join(exp_dir, f))])

        if num_files == 0:
            log.info(f"Number of Files: {num_files}. Skipping {exp_dir} due to no files.")
            continue  # Skip to the next iteration if there are no files

        # Proceed with processing if the directory exists and contains files
        try:
            novice_exp = ExpReader(exp_dir, verbose=True)
            novice_stroke_extr = StrokeExtractor(novice_exp)
            novice_stroke_metr = StrokeMetrics(novice_stroke_extr, num_buckets=5)
            
            # Define the output path for metrics and CSV files
            output_path = Path(output_base_dir) / os.path.basename(exp_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Save stroke metrics and bucket assignments to the output path
            novice_stroke_metr.save_stroke_metrics_and_buckets(output_path)
            
            # Generate the output path for the CSV file
            csv_output_path = output_path / "stroke_end_data.csv"
            
            # Save stroke ends to CSV
            stroke_ends = novice_stroke_extr._get_strokes(
                novice_exp.d_poses, 
                novice_exp.data_ts, 
                novice_stroke_extr.data_vrm_mask, 
                k=3, 
                output_csv_path=str(csv_output_path)
            )

            log.info(f"Stroke ends CSV saved to: {csv_output_path}")

            # Combine the stroke end data and stroke metrics
            # Load stroke_end_data.csv
            stroke_end_data = pd.read_csv(csv_output_path)

            # Load stroke_metrics.hdf5 (assuming it's located in the same directory)
            stroke_metrics_path = output_path / 'stroke_metrics.hdf5'
            with h5py.File(stroke_metrics_path, 'r') as f:
                stroke_lengths = f['length'][:]
                velocities = f['velocity'][:]
                accelerations = f['acceleration'][:]
                jerks = f['jerk'][:]
                curvatures = f['curvature'][:]
                voxels_removed = f['vxls_removed'][:]
                angle_wrt_camera = f['angle_wrt_camera'][:]

            # Create a DataFrame for stroke metrics with the valid fields
            stroke_metrics_df = pd.DataFrame({
                'Length': stroke_lengths,
                'Velocity': velocities,
                'Acceleration': accelerations,
                'Jerk': jerks,
                'Curvature': curvatures,
                'Voxels Removed': voxels_removed,
                'Angle with Respect to Camera': angle_wrt_camera,
            })

            # Combine stroke_end_data with stroke_metrics DataFrame
            combined_df = stroke_end_data.join(stroke_metrics_df)

            # Create a list to store the 'Stroke End Time' values
            time_values = []

            # Step 2: Populate the 'Stroke End Time' values list
            stroke_end_indices = combined_df.index[combined_df['Stroke_End'] == 1].tolist()

            if stroke_end_indices:
                # First occurrence where Stroke_End = 1
                first_stroke_index = stroke_end_indices[0]
                first_timestamp = combined_df.loc[first_stroke_index, 'Timestamp']
                initial_timestamp = combined_df.loc[0, 'Timestamp']

                # Calculate the time difference for the first occurrence
                time_values.append(first_timestamp - initial_timestamp)  # Append to the list

                # Loop through the remaining indices where Stroke_End = 1
                for i in range(1, len(stroke_end_indices)):
                    current_index = stroke_end_indices[i]
                    previous_index = stroke_end_indices[i - 1]

                    # Calculate time difference between consecutive Stroke_End = 1 occurrences
                    time_diff = combined_df.loc[current_index, 'Timestamp'] - combined_df.loc[previous_index, 'Timestamp']

                    # Add this difference to the previous "time" value and append to the list
                    time_values.append(time_values[-1] + time_diff)

            # Step 3: Convert the time_values list to a DataFrame
            time_df = pd.DataFrame(time_values, columns=['Stroke End Time'])

            # Step 4: Add the 'Stroke End Time' DataFrame back into 'combined_df' as a column
            combined_df['Stroke End Time'] = pd.Series(time_df['Stroke End Time'].values, index=combined_df.index[:len(time_df)])

            # Drop the first two columns and move "time" column to the first position
            combined_df_cleaned = combined_df.drop(combined_df.columns[:2], axis=1)
            columns = ['Stroke End Time'] + [col for col in combined_df_cleaned.columns if col != 'Stroke End Time']
            combined_df_cleaned = combined_df_cleaned[columns]

            # Generate the output path for the cleaned CSV file
            combined_csv_output_path = output_path / "timed_stroke_metrics.csv"
            combined_df_cleaned.to_csv(combined_csv_output_path, index=False)

            log.info(f"Cleaned stroke metrics saved to: {combined_csv_output_path}")

        except Exception as e:
            log.info(f"Error processing {exp_dir}: {e}")
        
if __name__ == "__main__":
    main()