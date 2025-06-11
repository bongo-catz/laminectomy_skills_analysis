import os
import pandas as pd
import argparse
import click 
import logging as log

# example usage: python3 scripts/merge_csv.py 
#                  --base-dir output_laminectomy/
#                  --output-file data/Laminectomy_Data/merged_extracted_metrics/combined_extracted_stroke_metrics.csv'

def load_valid_csv(folder_path, filename='timed_stroke_metrics.csv'):
    """
    Load and validate a CSV file from the given folder.

    Args:
        folder_path (str): Path to the folder containing the CSV.
        filename (str): Name of the CSV file to look for.

    Returns:
        pd.DataFrame or None: Cleaned DataFrame if valid, else None.
    """
    csv_file = os.path.join(folder_path, filename)
    if not os.path.exists(csv_file):
        return None

    df = pd.read_csv(csv_file)
    if df.empty or len(df.dropna(how='all', subset=df.columns.difference(['Source']))) <= 1:
        return None

    df['Source'] = os.path.basename(folder_path)
    df = df.dropna(how='all', subset=df.columns.difference(['Source']))
    return df

def collect_all_metrics(base_dir):
    """
    Collect stroke metrics from all valid subdirectories.

    Args:
        base_dir (str): Base directory containing subfolders with metric CSVs.

    Returns:
        pd.DataFrame: Combined DataFrame from all valid CSVs.
    """
    all_data = []

    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):
            df = load_valid_csv(folder_path)
            if df is not None:
                all_data.append(df)

    if not all_data:
        raise ValueError("No valid data found in the specified base directory.")

    return pd.concat(all_data, ignore_index=True)

@click.command()
@click.option('--base-dir', required=True, type=click.Path(exists=True),
              help='Path to the CSV file containing experiment directories.')
@click.option('--output-file', required=True, type=click.Path(),
              help='Path to save the combined CSV output.')
def main(base_dir, output_file):

    log.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=log.INFO)
    
    log.info(f"Reading data from: {base_dir}")
    log.info(f"Saving combined CSV to: {output_file}")

    try:
        combined_df = collect_all_metrics(base_dir)
        combined_df.to_csv(output_file, index=False)
        log.info("Combined stroke metrics saved successfully.")
    except Exception as e:
        log.info(f"❌ Error: {e}")


if __name__ == "__main__":
    main()