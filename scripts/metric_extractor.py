import os
import numpy as np
import pandas as pd
from collections import defaultdict
from exp_reader import ExpReader
from scipy import integrate
from scipy.spatial.transform import Rotation as R
from utils.hdf5_utils import save_dict_to_hdf5, load_dict_from_hdf5
from pathlib import Path
from plotting_skills_analysis import StrokeMetricsVisualizer, plot_3d_vx_rmvd
import h5py


# {short_name: [color, "long_name"]}
anatomy_dict = {
    "Bone": ["255 249 219", "Bone"],
    "Malleus": ["233 0 255", "Malleus"],
    "Incus": ["0 255 149", "Incus"],
    "Stapes": ["63 0 255", "Stapes"],
    "BonyLabyrinth": ["91 123 91", "Bony_Labyrinth"],
    "IAC": ["244 142 52", "IAC"],
    "SuperiorVestNerve": ["255 191 135", "Superior_Vestibular_Nerve"],
    "InferiorVestNerve": ["121 70 24", "Inferior_Vestibular_Nerve"],
    "CochlearNerve": ["219 244 52", "Cochlear_Nerve"],
    "FacialNerve": ["244 214 49", "Facial_Nerve"],
    "Chorda": ["151 131 29", "Chorda_Tympani"],
    "ICA": ["216 100 79", "ICA"],
    "SigSinus": ["110 184 209", "Sinus_+_Dura"],
    "VestAqueduct": ["91 98 123", "Vestibular_Aqueduct"],
    "TMJ": ["100 0 0", "TMJ"],
    "EAC": ["255 225 214", "EAC"],
}

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
            print(f"Stroke end data saved to {output_csv_path}")

        return stroke_ends

class StrokeMetrics:
    """
    A class to calculate various metrics related to strokes performed during a surgical procedure.

    Attributes:
    -----------
        exp : ExpReader object
            Created as synthesis o hdf5 files
        stroke_ends : np.ndarray
            1,0 array indicating the end points of strokes. Length of array is equal to data[data_vrm_mask]
        data_vrm_mask : np.ndarray
            Boolean mask for filtering drill pose data, to only drilling data
        stroke_endtimes : np.ndarray
            Array of timestamps representing when each stroke ends. Note that, the first timestamp
            is the minimum timestamp in drilling periods
        num_buckets : int
            Number of voxel buckets to categorize the strokes.
        bucket_dict : dict
            Dictionary assigning each stroke to a voxel bucket and defining bucket ranges.

    Methods:
    --------
        get_stroke_endtimes(stroke_ends, data_vrm_mask, data_ts):
            Returns an array of timestamps representing the end times of strokes.
        stroke_force(forces, forces_ts, stroke_endtimes):
            Computes the average force applied during each stroke.
        stroke_length(stroke_ends, d_poses, data_vrm_mask):
            Computes the length of each stroke.
        extract_kinematics(d_poses, data_ts, stroke_ends, data_vrm_mask):
            Extracts kinematic metrics (velocity, acceleration, jerk) for each stroke.
        extract_curvature(d_poses, data_ts, stroke_ends, data_vrm_mask):
            Calculates the curvature of each stroke.
        contact_orientation(stroke_endtimes, d_poses, data_ts, data_vrm_mask, forces, forces_ts):
            Computes the contact angle between the force and drill orientation for each stroke.
        orientation_wrt_camera(stroke_ends, stroke_endtimes, d_poses, cam_poses, data_ts, data_vrm_mask):
            Computes the average angle between the drill and camera for each stroke.
        voxels_removed(stroke_endtimes, v_rm_ts, v_rm_locs):
            Returns the number of voxels removed per stroke.
        gen_cum_vxl_rm_stroke_end(stroke_endtimes, v_rm_ts, v_rm_locs):
            Returns cumulative voxels removed at each stroke end time.
        calc_metrics():
            Computes function to actually calculate strokes using above methods
        assign_strokes_to_voxel_buckets():
            Assigns strokes to voxel buckets based on cumulative voxels removed.
        save_stroke_metrics_and_buckets(output_path):
            Saves the stroke metrics and voxel bucket assignments to HDF5 files.
        """
    
    def __init__(self, stroke_extr, num_buckets=5):
        """
        Initializes the StrokeMetrics class.

            Parameters:
                stroke_extr (object): An StrokeExtr object which gives us stroke segmentationa nd more
                num_buckets (int): The number of voxel buckets to categorize the strokes (default is 10).
        """
        self.exp = stroke_extr.exp
        self.stroke_ends = stroke_extr.stroke_ends
        self.data_vrm_mask = stroke_extr.data_vrm_mask
        self.stroke_endtimes = self.get_stroke_endtimes(self.stroke_ends, self.data_vrm_mask, self.exp.data_ts)
        self.num_buckets = num_buckets

        self.bucket_dict = self.assign_strokes_to_voxel_buckets()
        
        # Add this line to check if data is loaded correctly
        # print(f"Experiment data: {self.exp.d_poses}, {self.exp.data_ts}, {self.exp.v_rm_ts}")  

    def get_stroke_endtimes(self, stroke_ends, data_vrm_mask, data_ts):
        """
        Returns an array of timestamps representing the end times of strokes. Note that the first timestamp
        is the minimum timestamp in drilling periods.

            Parameters:
                stroke_ends (np.ndarray):  1,0 array indicating the end points of strokes. Length of array is equal to 
                    length of data[data_vrm_mask]
                data_vrm_mask (np.ndarray): Boolean mask for filtering drill pose data to only poses during drilling.
                data_ts (np.ndarray): Time stamps of all drill poses.

            Returns:
                stroke_endtimes (np.ndarray): Array of timestamps representing the end times of strokes, plus
                    first timestamp as the minimum timestamp in drilling periods.
        """        
        data_ts_vrm = data_ts[data_vrm_mask]
        stroke_endtimes = data_ts_vrm[stroke_ends.astype(bool)]
        stroke_endtimes = np.insert(stroke_endtimes, 0, min(data_ts_vrm))

        return stroke_endtimes       

    def stroke_force(self, forces, forces_ts, stroke_endtimes):
        """
        Computes the average force applied during each stroke.

        Note that force calculation doesnt work for some SDF Users Study Recordings, because 
        timestamps were recorded improperly. This is why we're getting the mean of an empty slice
        numpy errors.

            Parameters:
                forces (np.ndarray): Array of forces applied during the procedure. Sampled at higher
                    frequency than drill poses
                forces_ts (np.ndarray): Array of timestamps corresponding to each force.
                stroke_endtimes (np.ndarray): Array of timestamps representing the end times of strokes.

            Returns:
                avg_stroke_force (np.ndarray): Array of average forces applied during each stroke. Length
                    of array is equal to the number of strokes.
        """
        
        avg_stroke_force = []
        stroke_forces = 0
        
        forces = forces[:,:3] # excluding torques
        forces_ts = forces_ts[:forces.shape[0]] # Avoids mutex error, where extra timestamps are recorded

        
        for i in range(len(stroke_endtimes) - 1):
            stroke_mask = [f_ts >= stroke_endtimes[i] and f_ts < 
                           stroke_endtimes[i+1] for f_ts in forces_ts]
            stroke_forces = np.linalg.norm(forces[stroke_mask], axis=1) # Only considering magnitudes of forces here
            avg_stroke_force.append(np.mean(stroke_forces))
                
        return np.array(avg_stroke_force)
    
    def stroke_length(self, stroke_ends, d_poses, data_vrm_mask):
        """
        Computes the path length of each stroke.

            Parameters:
                stroke_ends (np.ndarray): 1,0 array indicating the end points of strokes. Length of array is equal to 
                    length of data[data_vrm_mask]
                d_poses (np.ndarray): Array of drill poses
                data_vrm_mask (np.ndarray): Boolean mask for filtering [data] group data to only drilling data

            Returns:
                lens (np.ndarray): Array of path lengths for each stroke. Length of array is equal to the number of strokes.
        """

        # note that this path length not euclidean length
        d_pos = d_poses[:, :3]
        d_pos = d_pos[data_vrm_mask]

        lens = []
        inds = np.insert(np.where(stroke_ends == 1), 0, 0)
        
        for i in range(sum(stroke_ends)):
            stroke_len = 0
            curr_stroke = d_pos[inds[i]:inds[i+1]]
            for j in range(1, len(curr_stroke)):
                stroke_len += np.linalg.norm(curr_stroke[j-1] - curr_stroke[j])

            lens.append(stroke_len)

        return np.array(lens)

    def extract_kinematics(self, d_poses, data_ts, stroke_ends, data_vrm_mask):
        """
        Extracts kinematic metrics (velocity, acceleration, jerk) for each stroke.

            Parameters:
                d_poses (np.ndarray): Array of drill poses
                data_ts (np.ndarray): Array of timestamps for each drill pose
                stroke_ends (np.ndarray): A 1,0 array indicating the end points of strokes. Length of array 
                    is equal to length of data[data_vrm_mask]
                data_vrm_mask (np.ndarray): Boolean mask for filtering [data] group data to only drilling data


            Returns:
                stroke_velocities (np.ndarray): Array of velocities for each stroke.
                stroke_accelerations (np.ndarray): Array of accelerations for each stroke.
                stroke_jerks (np.ndarray): Array of jerks for each stroke.
        """

        drill_pose = d_poses[data_vrm_mask]
        data_ts = data_ts[data_vrm_mask]
        stroke_indices = np.insert(np.where(stroke_ends == 1)[0] + 1, 0, 0)[:-1]
        
        # Extract x, y, and z data from drill pose data
        x = [i[0] for i in drill_pose]
        y = [i[1] for i in drill_pose]
        z = [i[2] for i in drill_pose]

        stroke_vx = []
        stroke_vy = []
        stroke_vz = []
        stroke_t = []
        stroke_velocities = []

        # Store velocity information for acceleration and calculate average velocity
        for i in range(len(stroke_indices)):
            stroke_start = stroke_indices[i]
            next_stroke = len(data_ts)
            if i != len(stroke_indices) - 1:
                next_stroke = stroke_indices[i + 1]
            # Split up positional data for each stroke
            stroke_x = x[stroke_start:next_stroke]
            stroke_y = y[stroke_start:next_stroke]
            stroke_z = z[stroke_start:next_stroke]
            t = data_ts[stroke_start:next_stroke]

            stroke_vx.append(np.gradient(stroke_x, t))
            stroke_vy.append(np.gradient(stroke_y, t))
            stroke_vz.append(np.gradient(stroke_z, t))
            stroke_t.append(t)

            # Calculate distance traveled during stroke and use to calculate velocity
            curr_stroke = [[stroke_x[k], stroke_y[k], stroke_z[k]]
                        for k in range(len(stroke_x))]
            dist = 0
            for l in range(1, len(curr_stroke)):
                dist += np.linalg.norm(np.subtract(
                    curr_stroke[l], curr_stroke[l - 1]))
            stroke_velocities.append(dist / np.ptp(t))

        # Calculate average acceleration using velocity information, and store for jerk
        stroke_accelerations = []
        for i in range(len(stroke_vx)):
            curr_stroke = [[stroke_vx[i][j], stroke_vy[i][j], stroke_vz[i][j]]
                        for j in range(len(stroke_vx[i]))]
            vel = 0
            for k in range(1, len(curr_stroke)):
                vel += np.linalg.norm(np.subtract(
                    curr_stroke[k], curr_stroke[k - 1]))
            stroke_accelerations.append(vel / np.ptp(stroke_t[i]))

        stroke_ax = []
        stroke_ay = []
        stroke_az = []

        # Store acceleration information for jerk
        for i in range(len(stroke_vx)):
            stroke_ax.append(np.gradient(stroke_vx[i], stroke_t[i]))
            stroke_ay.append(np.gradient(stroke_vy[i], stroke_t[i]))
            stroke_az.append(np.gradient(stroke_vz[i], stroke_t[i]))

        # Calculate average jerk using acceleration information
        stroke_jerks = []
        for i in range(len(stroke_ax)):
            curr_stroke = [[stroke_ax[i][j], stroke_ay[i][j], stroke_az[i][j]]
                        for j in range(len(stroke_ax[i]))]
            acc = 0
            for k in range(1, len(curr_stroke)):
                acc += np.linalg.norm(np.subtract(
                    curr_stroke[k], curr_stroke[k - 1]))
            stroke_jerks.append(acc / np.ptp(stroke_t[i]))
        

        return np.array(stroke_velocities), np.array(stroke_accelerations), np.array(stroke_jerks)

    def extract_curvature(self, d_poses, data_ts, stroke_ends, data_vrm_mask):
        """
        Calculates the curvature of each stroke according to definition by Rao et al. at
        https://doi.org/10.1023/A:1020350100748

            Parameters:
                d_poses (np.ndarray): Array of drill poses 
                data_ts (np.ndarray): Array of timestamps for each drill pose
                stroke_ends (np.ndarray): A1,0 array indicating the end points of strokes. Length of array 
                    is equal to length of data[data_vrm_mask]
                data_vrm_mask (np.ndarray): Boolean mask for filtering [data] group data to only drilling data

            Returns:
                curvatures (np.ndarray): Array of curvature values for each stroke.
        """

        drill_pose = d_poses[data_vrm_mask]
        data_ts = data_ts[data_vrm_mask]
        stroke_indices = np.insert(np.where(stroke_ends == 1)[0] + 1, 0, 0)[:-1]


        # Get preprocessed, x, y, and z position data
        x = [i[0] for i in drill_pose]
        y = [i[1] for i in drill_pose]
        z = [i[2] for i in drill_pose]

        curvatures = []
        stroke_curvatures = []
        for i in range(len(stroke_indices)):
            stroke_start = stroke_indices[i]
            next_stroke = len(data_ts)
            if i != len(stroke_indices) - 1:
                next_stroke = stroke_indices[i + 1]

            # Split up positional data for each stroke
            stroke_x = x[stroke_start:next_stroke]
            stroke_y = y[stroke_start:next_stroke]
            stroke_z = z[stroke_start:next_stroke]
            stroke_t = data_ts[stroke_start:next_stroke]

            # Calculate velocity and acceleration for each stroke
            stroke_vx = np.gradient(stroke_x, stroke_t)
            stroke_vy = np.gradient(stroke_y, stroke_t)
            stroke_vz = np.gradient(stroke_z, stroke_t)
            stroke_ax = np.gradient(stroke_vx, stroke_t)
            stroke_ay = np.gradient(stroke_vy, stroke_t)
            stroke_az = np.gradient(stroke_vz, stroke_t)
            curvature = []
            stroke_t_copy = [t for t in stroke_t]

            for j in range(len(stroke_vx)):
                # Calculate r' and r'' at specific time point
                r_prime = [stroke_vx[j], stroke_vy[j], stroke_vz[j]]
                r_dprime = [stroke_ax[j], stroke_ay[j], stroke_az[j]]

                # Potentially remove
                if np.linalg.norm(r_prime) == 0:
                    stroke_t_copy.pop(j)
                    continue
                k = np.linalg.norm(np.cross(r_prime, r_dprime)) / \
                    ((np.linalg.norm(r_prime)) ** 3)
                curvature.append(k)

            # Average value of function over an interval is integral of function divided by length of interval
            stroke_curvatures.append(integrate.simpson(
                curvature, stroke_t_copy) / np.ptp(stroke_t_copy))

        return np.array(stroke_curvatures)

    def contact_orientation(self, stroke_endtimes, d_poses, data_ts,
                        data_vrm_mask,forces, forces_ts):
        """
        Computes the contact angle between the bone normal vector and drill orientation for each stroke. We 
        use the force vector here to get the bone normal.

            Parameters:
                stroke_endtimes (np.ndarray): Array of timestamps representing the end times of strokes.
                d_poses (np.ndarray): Array of drill poses
                data_ts (np.ndarray): Array of timestamps for each drill pose
                data_vrm_mask (np.ndarray): Boolean mask for filtering [data] group data to only drilling data
                forces (np.ndarray): Array of forces applied during the procedure. Sampled at higher
                    frequency than drill poses
                forces_ts (np.ndarray): Array of timestamps corresponding to each force.
            
            Returns:
                avg_angles_per_stroke (np.ndarray): Array of average angles between the force and drill orientation
                    for each stroke.
        """

        d_poses = d_poses[data_vrm_mask]
        data_ts = data_ts[data_vrm_mask]
        forces = forces[:,:3] # Again excluding torques
        forces_ts = forces_ts[:forces.shape[0]] # Again avoiding mutex error
                              
        avg_angles_per_stroke = []

        for i in range(len(stroke_endtimes) - 1):
            stroke_mask = (forces_ts >= stroke_endtimes[i]) & (forces_ts < stroke_endtimes[i + 1])
            stroke_forces = forces[stroke_mask]
            stroke_forces_ts = forces_ts[stroke_mask]

            if stroke_forces.size == 0:
                avg_angles_per_stroke.append(np.nan)  # Append NaN for strokes with no data, unlikely tho
                continue

            # Getting drill poses closest to force timestamps
            closest_indices = np.abs(np.subtract.outer(data_ts, stroke_forces_ts)).argmin(axis=0)
            closest_d_poses = d_poses[closest_indices]

            stroke_angles = []

            for force, pose in zip(stroke_forces, closest_d_poses):
                drill_vec =  R.from_quat(pose[3:]).apply([-1, 0, 0]) # assuming the drill is pointing in the x direction
                drill_vec = drill_vec / np.linalg.norm(drill_vec)
                
                force_norm = np.linalg.norm(force)
                if force_norm <= 0:
                    continue
                normal = force / force_norm
                
                angle = np.arccos(np.clip(np.dot(normal, drill_vec), -1.0, 1.0)) * (180 / np.pi)
                if angle > 90:
                    angle = 180 - angle

                stroke_angles.append(90 - angle)
            
            # Taking average of angles over each stroke
            avg = np.mean(stroke_angles)
            avg_angles_per_stroke.append(avg)
        
        return np.array(avg_angles_per_stroke)

    def orientation_wrt_camera(self, stroke_ends, stroke_endtimes, d_poses, cam_poses, data_ts, data_vrm_mask):
        """
        Computes the average angle between the drill and camera for each stroke.

            Parameters:
                stroke_ends (np.ndarray): A 1,0 array indicating the end points of strokes. Length of array 
                    is equal to length of data[data_vrm_mask]
                stroke_endtimes (np.ndarray): Array of timestamps representing the end times of strokes.
                d_poses (np.ndarray): Array of drill poses
                cam_poses (np.ndarray): Array of camera poses
                data_ts (np.ndarray): Array of timestamps for each drill pose
                data_vrm_mask (np.ndarray): Boolean mask for filtering [data] group data to only drilling data

            Returns:
                avg_angle_wrt_camera (np.ndarray): Array of average angles between the drill and camera for each stroke.
        """
        d_poses = d_poses[data_vrm_mask]
        data_ts = data_ts[data_vrm_mask]
        cam_poses = cam_poses[data_vrm_mask]

        avg_angles = []

        for i in range(len(stroke_endtimes) - 1):  # Ensuring stroke_ends[i+1] is valid
            stroke_mask = (data_ts >= stroke_endtimes[i]) & (data_ts < stroke_endtimes[i+1])
            stroke_poses = d_poses[stroke_mask]
            stroke_cam_poses = cam_poses[stroke_mask]

            if len(stroke_poses) == 0:
                avg_angles.append(np.nan)
                continue

            # Calculate orientations
            drill_quats = stroke_poses[:, 3:]
            cam_quats = stroke_cam_poses[:, 3:]

            drill_rot = R.from_quat(drill_quats).apply([-1, 0, 0])
            cam_rot = R.from_quat(cam_quats).apply([-1, 0, 0])

            # Normalize
            drill_rot = drill_rot / np.linalg.norm(drill_rot, axis=1)[:, np.newaxis]
            cam_rot = cam_rot / np.linalg.norm(cam_rot, axis=1)[:, np.newaxis]

            # Calculate angles between vectors
            angles = np.arccos(np.clip(np.einsum('ij,ij->i', drill_rot, cam_rot), -1.0, 1.0)) * (180 / np.pi)
            angles = np.where(angles > 90, 180 - angles, angles)
            avg_angle = np.mean(90 - angles)

            avg_angles.append(avg_angle)

        avg_angle_wrt_camera = np.array(avg_angles)

        return avg_angle_wrt_camera

    def gen_cum_vxl_rm_stroke_end(self, stroke_endtimes, v_rm_ts, v_rm_locs):
        """
        Generates cumulative voxels removed at each stroke end time. Length of array is equal to the number of strokes.

            Parameters:
                stroke_endtimes (np.ndarray): Array of timestamps representing the end times of strokes.
                v_rm_ts (np.ndarray): Array of timestamps for each voxel removed.
                v_rm_locs (np.ndarray): Array of voxel removed locations.
            
            Returns:
                cum_vxl_rm_stroke_end (np.ndarray): Array of cumulative voxels removed at each stroke end time.
        """
        stroke_endtimes = stroke_endtimes[1:] # remove first timestamp which is min(v_rm_ts) and not needed
        closest_vxl_rm_ts_to_stroke_end = np.searchsorted(v_rm_ts, stroke_endtimes)
        cum_vxl_rm_stroke_end = np.searchsorted(v_rm_locs[:,0], closest_vxl_rm_ts_to_stroke_end, side='right') -1 
        
        return np.array(cum_vxl_rm_stroke_end)
    
    def voxels_removed(self, stroke_endtimes, v_rm_ts, v_rm_locs):
        """
        Returns the number of voxels removed per stroke.

            Parameters:
                stroke_endtimes (np.ndarray): Array of timestamps representing the end times of strokes.
                v_rm_ts (np.ndarray): Array of timestamps for each voxel removed.
                v_rm_locs (np.ndarray): Array of voxel removed locations.
            
            Returns:
                vxls_removed (np.ndarray): Array of number of voxels removed per stroke.
        """
        cum_vxl_rm_stroke_end = self.gen_cum_vxl_rm_stroke_end(stroke_endtimes, v_rm_ts, v_rm_locs)
        vxls_removed = np.diff(cum_vxl_rm_stroke_end, prepend=0)

        return vxls_removed
    
    def calc_metrics (self):
        """
        Compute function to actually calculate stroke metrics using above methods. All storke metrics 
        are of equal length, equal to the number of strokes.

            Returns:
                metrics_dict (dict): Dictionary of all stroke metrics.
        """
        length = self.stroke_length(self.stroke_ends, self.exp.d_poses, self.data_vrm_mask)
        velocity, acceleration, jerk = self.extract_kinematics(self.exp.d_poses, self.exp.data_ts, self.stroke_ends, self.data_vrm_mask)
        curvature = self.extract_curvature(self.exp.d_poses, self.exp.data_ts, self.stroke_ends, self.data_vrm_mask)
        orientation_wrt_camera = self.orientation_wrt_camera(self.stroke_ends, self.stroke_endtimes, self.exp.d_poses, self.exp.cam_poses, self.exp.data_ts, self.data_vrm_mask)
        voxels_removed = self.voxels_removed(self.stroke_endtimes, self.exp.v_rm_ts, self.exp.v_rm_locs)

        force = self.stroke_force(self.exp.forces, self.exp.forces_ts, self.stroke_endtimes)
        contact_angle = self.contact_orientation(self.stroke_endtimes, self.exp.d_poses, self.exp.data_ts, self.data_vrm_mask, self.exp.forces, self.exp.forces_ts)
        
        if not sum(np.isnan(force)) > 0.5*len(force):
            metrics_dict = {'length': length, 'velocity': velocity, 'acceleration': acceleration, 'jerk': jerk, 'vxls_removed': voxels_removed, 
                            'curvature': curvature, 'force': force, 'angle_wrt_bone': contact_angle, 'angle_wrt_camera': orientation_wrt_camera}
        else:
            metrics_dict = {'length': length, 'velocity': velocity, 'acceleration': acceleration, 'jerk': jerk, 'vxls_removed': voxels_removed,
                            'curvature': curvature, 'angle_wrt_camera': orientation_wrt_camera}
       
        # print(f"Metrics calculated: {metrics_dict}") # for checking if metrics are being calculated
        return metrics_dict

    def assign_strokes_to_voxel_buckets(self):
        """
        Creates the user specified number of buckets, by dividing the total voxels removed equally amongst all buckets.
        Then assigns each stroke to a bucket based on the cumulative voxels removed at the end of each stroke.

            Returns:
                bucket_dict (dict): Dictionary with two keys, 'bucket_assignments' and 'bucket_ranges'. 
                    'bucket_assignments' is a list of bucket assignments for each stroke. 'bucket_ranges' is a list of tuples
                    representing the range of each bucket.  
        """
        num_buckets = self.num_buckets

        # Generate cumulative voxels removed at stroke ends
        cum_vxl_rm_stroke_end = self.gen_cum_vxl_rm_stroke_end(self.stroke_endtimes, self.exp.v_rm_ts, self.exp.v_rm_locs)
        total_voxels = cum_vxl_rm_stroke_end[-1]
        
        # Determine the range of each bucket
        bucket_size = total_voxels / num_buckets
        bucket_ranges = [(int(i * bucket_size), int((i + 1) * bucket_size - 1)) for i in range(num_buckets)]
        
        # Assign each stroke to a bucket
        bucket_assignments = np.zeros(len(cum_vxl_rm_stroke_end), dtype=int)
        
        for i, voxel_count in enumerate(cum_vxl_rm_stroke_end):
            # Find the bucket index; max is to handle the last bucket edge case
            bucket_index = min(int(voxel_count / bucket_size), num_buckets - 1)
            bucket_assignments[i] = bucket_index


        bucket_dict = {'bucket_assignments': bucket_assignments, 'bucket_ranges': bucket_ranges}  
        
        return bucket_dict
    
    def save_stroke_metrics_and_buckets(self, output_path):
        """
        Saves the stroke metrics and voxel bucket dict to HDF5 files.
        """
        metrics = self.calc_metrics()
        bucket_dict = self.assign_strokes_to_voxel_buckets()

        save_dict_to_hdf5(metrics, output_path / 'stroke_metrics.hdf5')
        save_dict_to_hdf5(bucket_dict, output_path / 'stroke_buckets.hdf5')

        return metrics, bucket_dict

class GenMetrics:
    """
    Object to calculate general metrics for each experiment

    Attributes:
    -----------
        stroke_extr : StrokeExtr object
            An object that gives us stroke segmentation and more
        exp_dir : str
            The directory of the experiment
    
    Methods:
        procedure_time(): Returns the total time of the procedure as a float
        num_strokes(): Returns the total number of strokes as an int
        metadata_dict(): Returns metadata dictionary containing participant name, volume name, assist mode, and trial number.
        voxel_rmvd_dict(): Returns a dictionary with the number of voxels removed (value) for each anatomy (key).
        burr_change_dict(threshold = 0.8): Returns dictionary with burr size, and percent time spent in each burr size
        calc_metrics(): Actually calculates all the gen metrics and returns a dictionary with the procedure time,
            number of strokes, metadata, voxels removed, and burr changes.
        save_gen_metrics(output_path): Saves the general metrics to a HDF5 file.
    """
    def __init__(self, stroke_extr, exp_dir):
        """
        Initializes the GenMetrics class.

            Parameters:
                stroke_extr (object): An StrokeExtr object which gives us stroke segmentation and more
                exp_dir (str): The directory of the experiment
        """
        # Not sure if this lazy intiializaiton with a stroke_extr object is good practice
        self.exp = stroke_extr.exp
        self.stroke_ends = stroke_extr.stroke_ends
        self.data_vrm_mask = stroke_extr.data_vrm_mask
        self.exp_dir = exp_dir
    
    def procedure_time(self):
        """
        Returns the total time of the procedure as a float
        """
        # Copy of method from StrokeMetrics TODO: remove redundancy
        data_ts_vrm = self.exp.data_ts[self.data_vrm_mask]
        stroke_endtimes = data_ts_vrm[self.stroke_ends.astype(bool)]
        stroke_endtimes = np.insert(stroke_endtimes, 0, min(data_ts_vrm))
        self.stroke_endtimes = stroke_endtimes

        return stroke_endtimes[-1] - stroke_endtimes[0]
    
    def num_strokes(self):
        """
        Returns the total number of strokes as an int
        """

        return sum(self.stroke_ends)
    
    def metadata_dict(self):
        """
        Returns metadata dictionary containing participant name, volume name, assist mode, and trial number.        
        """

        with open(self.exp_dir + '/metadata.json', 'r') as f:
            metadata = f.read()

        return metadata
    
    def voxel_rmvd_dict(self):
        """
        Returns a dictionary with the number of voxels removed (value) for each anatomy (key).
        """
        # Returns a dictionary with the number of voxels removed (value) for each anatomy (key)
        vxl_rmvd_dict = defaultdict(int)
        v_rm_colors = np.array(self.exp.v_rm_colors).astype(np.int32)
        v_rm_colors_df = pd.DataFrame(v_rm_colors, columns=["ts_idx", "r", "g", "b", "a"])

        # add a column with the anatomy names
        for name, anatomy_info_list in anatomy_dict.items():
            color, full_name = anatomy_info_list
            color = list(map(int, color.split(" ")))
            v_rm_colors_df.loc[
                (v_rm_colors_df["r"] == color[0])
                & (v_rm_colors_df["g"] == color[1])
                & (v_rm_colors_df["b"] == color[2]),
                "anatomy_name",
            ] = name
        
        # Count number of removed voxels of each anatomy
        voxel_summary = v_rm_colors_df.groupby(["anatomy_name"]).count()

        for anatomy in voxel_summary.index:
            vxl_rmvd_dict[anatomy] += voxel_summary.loc[anatomy, "a"]
        
        return dict(vxl_rmvd_dict)
    
    def burr_change_dict(self, threshold = 0.8):
        
        """
        Returns dictionary with burr size, and percent time spent in each burr size
        #TODO change to percent voxel removed in each burr size
        """
        if self.exp.burr_chg_sz is None:
            burr_chg_dict = {'6 mm': 1.0}
            return burr_chg_dict

        burr_chg_sz = np.array(self.exp.burr_chg_sz)
        burr_chg_ts = np.array(self.exp.burr_chg_ts)

        burr_chg_sz = np.insert(burr_chg_sz, 0, 6) # 6 is starting burr size
        burr_chg_ts = np.append(burr_chg_ts, self.stroke_endtimes[-1]) # changing time stamps to represent the time at which the burr size changes from corresponding size (not to) 

    
        # calculate differences between consecutive changes
        diffs = np.diff(burr_chg_ts)
        diffs = np.append(diffs, True) # keep last change

        # select elements where the difference is >= 0.8s
        burr_chg_sz = burr_chg_sz[diffs >= threshold]
        burr_chg_ts = burr_chg_ts[diffs >= threshold]

        burr_sz_duration = np.diff(burr_chg_ts, prepend=self.stroke_endtimes[0])
        relative_burr_duration = burr_sz_duration / self.procedure_time()


        burr_chg_dict = {str(burr_size) + ' mm': 0 for burr_size in np.unique(burr_chg_sz)}
        for i in range(len(burr_chg_ts)):
            burr_size_str = str(burr_chg_sz[i]) + ' mm'
            burr_chg_dict[burr_size_str] += relative_burr_duration[i]
        
        return dict(burr_chg_dict)
    
    def calc_metrics(self):
        """
        Actually calculates all the gen metrics and returns a dictionary with the procedure time, 
        number of strokes, metadata, voxels removed, and burr changes.  
        """
        procedure_time = self.procedure_time()
        num_strokes = self.num_strokes()
        metadata = self.metadata_dict()
        vxl_rmvd_dict = self.voxel_rmvd_dict()
        burr_chg_dict = self.burr_change_dict()

        self.gen_metrics_dict = {'procedure_time': procedure_time, 'num_strokes': num_strokes, 'metadata': metadata, 'vxl_rmvd_dict': vxl_rmvd_dict, 'burr_chg_dict': burr_chg_dict}

        return self.gen_metrics_dict
    
    def save_gen_metrics(self):
        """
        Saves the gen metrics to an HDF5 file. 
        """

        save_dict_to_hdf5(self.gen_metrics_dict, self.exp_dir + '/gen_metrics.hdf5')
        


def main():
    exp_csv = pd.read_csv('/Users/orenw/Documents/tb_skills_analysis/data/SDF_UserStudy_Data/exp_dirs_DONOTOVERWRITE.csv')
    exps = exp_csv['exp_dir']

    for i in range(len(exps)):
        exp_dir = exps[i]
        print(f"Processing: {exp_dir}")

        # Check if the directory exists
        if not os.path.exists(exp_dir):
            print(f"Directory does not exist: {exp_dir}. Skipping.")
            continue  # Skip to the next iteration if the directory doesn't exist

        # Check if there are any files in the experiment directory
        num_files = len([f for f in os.listdir(exp_dir) if os.path.isfile(os.path.join(exp_dir, f))])

        if num_files == 0:
            print(f"Number of Files: {num_files}. Skipping {exp_dir} due to no files.")
            continue  # Skip to the next iteration if there are no files

        # Proceed with processing if the directory exists and contains files
        try:
            novice_exp = ExpReader(exp_dir, verbose=True)
            novice_stroke_extr = StrokeExtractor(novice_exp)
            novice_stroke_metr = StrokeMetrics(novice_stroke_extr, num_buckets=5)
            
            # Define the output path for metrics and CSV files
            output_path = Path(f"/Users/orenw/Documents/tb_skills_analysis/data/SDF_UserStudy_Data/extracted_metrics/{os.path.basename(exp_dir)}")
            output_path.mkdir(parents=True, exist_ok=True)  # Create the output directory if it doesn't exist

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

            print(f"Stroke ends CSV saved to: {csv_output_path}")

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

            print(f"Cleaned stroke metrics saved to: {combined_csv_output_path}")

        except Exception as e:
            print(f"Error processing {exp_dir}: {e}")



    # att_exp  = ExpReader(exps[93], verbose = True)
    # att_stroke_extr = StrokeExtractor(att_exp)
    # att_stroke_metr = StrokeMetrics(att_stroke_extr, num_buckets = 5)
    # att_metrics_dict = att_stroke_metr.calc_metrics()
    # att_bucket_dict = att_stroke_metr.assign_strokes_to_voxel_buckets()

    
    # visualizer = StrokeMetricsVisualizer(novice_metrics_dict, novice_bucket_dict, att_metrics_dict, att_bucket_dict, plot_previous_bucket=False)
    # visualizer.interactive_plot_buckets() 

    # novice_gen_metr = GenMetrics(novice_stroke_extr, exps[46])
    # plot_3d_vx_rmvd(novice_exp)
    

    # att_exp = exp_reader(exps[46], verbose = True)
    # att_stroke_extr = stroke_extractor(att_exp)
    # att_stroke_metr = stroke_metrics(att_exp, att_stroke_extr.stroke_ends, att_stroke_extr.data_vrm_mask)
    # att_metrics_dict = att_stroke_metr.calc_metrics()

    # stroke_metr.plot_metrics(metrics_dict)
    # novice_stroke_metr.interactive_plot(window_percentage=30, overlap_percentage=80)
        
    # break
        

if __name__ == "__main__":
    main()