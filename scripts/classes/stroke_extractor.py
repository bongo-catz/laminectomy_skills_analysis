# classes/stroke_extractor.py
import numpy as np
import pandas as pd
import logging as log
from scipy.spatial.transform import Rotation as R

class StrokeExtractor:
    """
    A class to extract stroke segmentation from an experiment

    Attributes:
    -----------
        exp : ExpReader object
            Created as synthesis of hdf5 files
        stroke_ends : np.ndarray
            1,0 array indicating the end points of strokes. Length of array is equal to data[data_vrm_mask]
        data_vrm_mask : np.ndarray
            Boolean mask for filtering [data] group data, to only drilling data

    Methods:
    -----------
        _find_drilling_seq(v_rm_ts, threshold=0.2):
            Find sequences of time points where each consecutive pair is less than `threshold` seconds.
        _filter_pose_ts_within_periods(data_ts, v_rm_ts):
            Filters and retains time points that fall within any of the specified time periods.
        _get_strokes(d_poses, data_ts, data_vrm_mask, k=3):
            Returns a list of 1's and 0's indicating whether a stroke has ended at the timestamp at its index and the timestamps.

    """
    def __init__(self, exp):
        """
        Initializes the StrokeExtractor class.

            Parameters:
                - exp (ExpReader): An ExpReader object which gives us the experiment data.

        """
        self.exp = exp

        self.data_vrm_mask  = self._filter_pose_ts_within_periods(self.exp.data_ts, self.exp.v_rm_ts)
        self.stroke_ends = self._get_strokes(self.exp.d_poses, self.exp.data_ts, self.data_vrm_mask)
        
    def _find_drilling_seq(self, v_rm_ts, threshold=0.2):
        """
        Find sequences of time points where each consecutive pair is less than `threshold` seconds. This is
        generally to find periods of drilling in the data, as we're finding sequences in the voxels removed
        dataset

        Parameters:
        - time_points: A sorted NumPy array of time points.
        - threshold (seconds): The maximum allowed difference between consecutive time points to consider them a sequence.

        Returns:
        - A list of tuples, where each tuple contains the start and end of a sequence.
        """

        # Calculate differences between consecutive time points
        diffs = np.diff(v_rm_ts)

        # Identify where differences are greater than or equal to the threshold
        breaks = np.where(diffs >= threshold)[0]

        # Calculate start and end indices for sequences
        starts = np.concatenate(([0], breaks + 1))
        ends = np.concatenate((breaks, [len(v_rm_ts) - 1]))

        # Filter out sequences where the start and end are the same (i.e., no actual sequence)
        drill_periods = [(v_rm_ts[start], v_rm_ts[end]) for start, end in zip(starts, ends) if start != end]

        return drill_periods

    def _filter_pose_ts_within_periods(self, data_ts, v_rm_ts):
        """
        Filters and retains time points that fall within any of the specified time periods. Goal is to filter
        out drill pose data points within drilling periods

        Parameters:
        - time_points: A numpy array of time points.
        - periods: A list of tuples, where each tuple contains a start and end time point of a period.

        Returns:
        - Boolean mask of driving the time points that fall within any of the specified periods.
        """
        periods = self._find_drilling_seq(v_rm_ts)

        # Initialize an empty array to mark time points within periods
        is_within_period = np.zeros(data_ts.shape, dtype=bool)

        # Iterate through each period and mark time points within the period
        for start, end in periods:
            is_within_period |= (data_ts >= start) & (data_ts <= end)

        return is_within_period

    def _get_strokes(self, d_poses, data_ts, data_vrm_mask, k=3, output_csv_path=None):
        '''
        Returns a list of 1's and 0's indicating whether a stroke has
        ended at the timestamp at its index and the timestamps.
        Optionally, saves the results to a CSV file.

        Parameters:
            d_poses (np.ndarray): Drill poses over course of procedure
            data_ts (np.ndarray): Time stamps of all drill poses
            data_vrm_mask (np.ndarray): Mask for filtering drill data
            k (int): Number of neighboring points for computing k-cosines
            output_csv_path (str): Optional. Path to save the CSV file.

        Returns:
            np.ndarray: List of 1's and 0's indicating whether a stroke has ended.
        '''
        # Add validation
        assert len(d_poses) == len(data_ts), "d_poses and data_ts must have same length"
        assert len(data_ts) == len(data_vrm_mask), "data_ts and data_vrm_mask must have same length"
        d_pos = d_poses[data_vrm_mask][:,:3]
        data_ts = data_ts[data_vrm_mask]
        K_p = []

        # Compute k-cosines for each pivot point
        for pivot, P in enumerate(d_pos):
            if (pivot - k < 0) or (pivot + k >= d_pos.shape[0]):
                continue

            # Get vector of previous k points and pivot and after k points and pivot
            v_bf = d_pos[pivot] - d_pos[pivot - k]
            v_af = d_pos[pivot] - d_pos[pivot + k]
            cos_theta = np.dot(v_bf, v_af) / (np.linalg.norm(v_bf) * np.linalg.norm(v_af))
            theta = np.arccos(cos_theta) * 180 / np.pi
            K = 180 - theta # now in degrees
            K_p.append(K)
        
        # Detect pivot points as points with k-cosines greater than mu + sig
        mu = np.mean(K_p)
        sig = np.std(K_p)

        for i in range(k):
            K_p.insert(0, mu)
            K_p.append(mu)
        
        stroke_ends = [1 if k_P > mu + sig else 0 for k_P in K_p]
        stroke_ends = np.array(stroke_ends)
        
        # Calculate speeds as tiebreak for consecutive pivot points
        position_diffs = np.diff(d_pos, axis=0)
        dists = np.linalg.norm(position_diffs, axis=1)
        time_diffs = np.diff(data_ts)
        speeds = dists / time_diffs
        speeds = np.insert(speeds, 0, 0)

        pivot_props = np.where(stroke_ends == 1)[0]  # Indices where stroke_ends == 1
        
        # Iterate over pivot_props and eliminate consecutive ones
        i = 0
        while i < len(pivot_props) - 1:
            if pivot_props[i] + 1 == pivot_props[i + 1]:
                start = i
                while i < len(pivot_props) - 1 and pivot_props[i] + 1 == pivot_props[i + 1]:
                    i += 1
                end = i
                min_val_index = np.argmin(speeds[pivot_props[start:end + 1]])
                for j in range(start, end + 1):
                    if j != start + min_val_index:
                        stroke_ends[pivot_props[j]] = 0
            i += 1
            
        stroke_ends[len(stroke_ends)-1] = 1
        
        # Optionally save the result to a CSV file
        if output_csv_path:
            # Create a DataFrame to store the timestamps and stroke ends
            stroke_data = pd.DataFrame({
                'Timestamp': data_ts,
                'Stroke_End': stroke_ends
            })
            stroke_data.to_csv(output_csv_path, index=False)
            log.info(f"Stroke end data saved to {output_csv_path}")

        return stroke_ends