import h5py
import numpy as np
from plotting_skills_analysis import StrokeMetricsVisualizer

def load_hdf5_file(file_path):
    """
    Loads an HDF5 file into a Python dictionary.
    """
    data = {}
    with h5py.File(file_path, 'r') as f:
        for key in f.keys():
            data[key] = np.array(f[key])
    return data

# Define the paths to your saved metrics_dict and bucket_dict
metrics_dict_path = "/Users/orenw/Documents/tb_skills_analysis/data/SDF_UserStudy_Data/extracted_metrics/2023-02-10 09:30:27_anatC_visual_P9T1/stroke_metrics.hdf5"
bucket_dict_path = "/Users/orenw/Documents/tb_skills_analysis/data/SDF_UserStudy_Data/extracted_metrics/2023-02-10 09:30:27_anatC_visual_P9T1/stroke_buckets.hdf5"

# Load the dictionaries
metrics_dict = load_hdf5_file(metrics_dict_path)
bucket_dict = load_hdf5_file(bucket_dict_path)

# Initialize and run the visualizer
visualizer = StrokeMetricsVisualizer(metrics_dict, bucket_dict)

# Show histograms
visualizer.interactive_plot_buckets()
