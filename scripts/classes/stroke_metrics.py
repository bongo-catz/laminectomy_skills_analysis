# classes/stroke_metrics.py
import numpy as np
from scipy import integrate
from scipy.spatial.transform import Rotation as R
from utils.hdf5_utils import save_dict_to_hdf5
import logging as log
import pandas as pd

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
        
        # Add voxel color data if available, ensure voxel colors and timestamps are aligned
        if hasattr(stroke_extr.exp, 'v_rm_colors') and hasattr(stroke_extr.exp, 'v_rm_ts'):
            min_len = min(len(stroke_extr.exp.v_rm_colors), len(stroke_extr.exp.v_rm_ts))
            self.v_rm_colors = stroke_extr.exp.v_rm_colors[:min_len]
            # Make sure we use the same length for timestamps
            self.exp.v_rm_ts = self.exp.v_rm_ts[:min_len]
        else:
            self.v_rm_colors = None
        
        self.bucket_dict = self.assign_strokes_to_voxel_buckets()
        # Add this line to check if data is loaded correctly
        # log.info(f"Experiment data: {self.exp.d_poses}, {self.exp.data_ts}, {self.exp.v_rm_ts}")  

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
        
        #forces = forces[:,:3] # excluding torques
        #forces_ts = forces_ts[:forces.shape[0]] # Avoids mutex error, where extra timestamps are recorded
        # Safety: ensure alignment
        forces = forces[:,:3]  # use only linear forces
        min_len = min(len(forces), len(forces_ts))
        forces = forces[:min_len]
        forces_ts = forces_ts[:min_len]
        
        for i in range(len(stroke_endtimes) - 1):
            # stroke_mask = [f_ts >= stroke_endtimes[i] and f_ts < 
            #                stroke_endtimes[i+1] for f_ts in forces_ts]
            stroke_mask = (forces_ts >= stroke_endtimes[i]) & (forces_ts < stroke_endtimes[i+1])
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
        
        # forces = forces[:,:3] # Again excluding torques
        # forces_ts = forces_ts[:forces.shape[0]] # Again avoiding mutex error
        # Safety: ensure alignment
        forces = forces[:,:3]  # use only linear forces
        min_len = min(len(forces), len(forces_ts))
        forces = forces[:min_len]
        forces_ts = forces_ts[:min_len]
                                
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
        log.info("Extracting stroke length...")
        length = self.stroke_length(self.stroke_ends, self.exp.d_poses, self.data_vrm_mask)
        log.info("Extracting stroke kinematics...")
        velocity, acceleration, jerk = self.extract_kinematics(self.exp.d_poses, self.exp.data_ts, self.stroke_ends, self.data_vrm_mask)
        log.info("Extracting stroke curvature...")
        curvature = self.extract_curvature(self.exp.d_poses, self.exp.data_ts, self.stroke_ends, self.data_vrm_mask)
        log.info("Extracting orientation wrt camera...")
        orientation_wrt_camera = self.orientation_wrt_camera(self.stroke_ends, self.stroke_endtimes, self.exp.d_poses, self.exp.cam_poses, self.exp.data_ts, self.data_vrm_mask)
        log.info("Extracting voxels removed per stroke...")
        voxels_removed = self.voxels_removed(self.stroke_endtimes, self.exp.v_rm_ts, self.exp.v_rm_locs)

        log.info("Extracting stroke forces...")
        force = self.stroke_force(self.exp.forces, self.exp.forces_ts, self.stroke_endtimes)
        log.info("Extracting stroke contact orientation...")
        contact_angle = self.contact_orientation(self.stroke_endtimes, self.exp.d_poses, self.exp.data_ts, self.data_vrm_mask, self.exp.forces, self.exp.forces_ts)
        
        # Calculate color breakdown if available
        if self.v_rm_colors is not None:
            log.info("Calculating voxel color breakdown...")
            color_breakdowns = [self.compute_voxel_color_breakdown(i) for i in range(len(self.stroke_endtimes)-1)]
            green_pct = [bd["Green"] for bd in color_breakdowns]
            yellow_pct = [bd["Yellow"] for bd in color_breakdowns] 
            red_pct = [bd["Red"] for bd in color_breakdowns]
            other_pct = [bd["Other"] for bd in color_breakdowns]
        else:
            green_pct = yellow_pct = red_pct = other_pct = np.zeros(len(self.stroke_endtimes)-1)
        
        if not sum(np.isnan(force)) > 0.5*len(force):
            metrics_dict = {'length': length, 'velocity': velocity, 'acceleration': acceleration, 'jerk': jerk, 'vxls_removed': voxels_removed, 
                            'curvature': curvature, 'force': force, 'angle_wrt_bone': contact_angle, 'angle_wrt_camera': orientation_wrt_camera,
                            'voxel_pct_green': green_pct, 'voxel_pct_yellow': yellow_pct, 'voxel_pct_red': red_pct, 'voxel_pct_other': other_pct,
                            'voxels_removed_rgba': self.v_rm_colors if self.v_rm_colors is not None else np.array([])}
        else:
            metrics_dict = {'length': length, 'velocity': velocity, 'acceleration': acceleration, 'jerk': jerk, 'vxls_removed': voxels_removed,
                            'curvature': curvature, 'angle_wrt_camera': orientation_wrt_camera, 'voxel_pct_green': green_pct, 'voxel_pct_yellow': yellow_pct,
                            'voxel_pct_red': red_pct, 'voxel_pct_other': other_pct, 'voxels_removed_rgba': self.v_rm_colors if self.v_rm_colors is not None else np.array([])}
       
        # log.info(f"Metrics calculated: {metrics_dict}") # for checking if metrics are being calculated
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
        log.info("Calculating metrics for strokes...")
        metrics = self.calc_metrics()
        log.info("Assigning strokes to voxel buckets...")
        bucket_dict = self.assign_strokes_to_voxel_buckets()

        save_dict_to_hdf5(metrics, output_path / 'stroke_metrics.hdf5')
        save_dict_to_hdf5(bucket_dict, output_path / 'stroke_buckets.hdf5')

        return metrics, bucket_dict
    
    @staticmethod
    def classify_voxel_color(alpha_adjusted):
        """Classify voxel color based on normalized alpha value."""
        if alpha_adjusted == 1.0:
            return "Green"
        elif alpha_adjusted == 0.3:
            return "Yellow"
        elif alpha_adjusted in [0.1, 0.2]:
            return "Red"
        else:
            return "Other"

    def compute_voxel_color_breakdown(self, stroke_idx):
        """
        Computes percentage breakdown of voxel colors removed in a stroke.
        
        Args:
            stroke_idx (int): Index of the stroke to analyze
            
        Returns:
            dict: Dictionary with percentage of each color category
        """
        if self.v_rm_colors is None:
            return {"Green": 0, "Yellow": 0, "Red": 0, "Other": 100}
            
        # Get voxel timestamps for this stroke
        start_time = self.stroke_endtimes[stroke_idx]
        end_time = self.stroke_endtimes[stroke_idx + 1]
        
        # Find voxels removed during this stroke
        stroke_mask = (self.exp.v_rm_ts >= start_time) & (self.exp.v_rm_ts < end_time)
        stroke_voxels = self.v_rm_colors[stroke_mask]
        
        if len(self.exp.v_rm_ts) != len(self.v_rm_colors):
            min_len = min(len(self.exp.v_rm_ts), len(self.v_rm_colors))
            stroke_mask = (self.exp.v_rm_ts[:min_len] >= start_time) & (self.exp.v_rm_ts[:min_len] < end_time)
            stroke_voxels = self.v_rm_colors[:min_len][stroke_mask]
        else:
            stroke_mask = (self.exp.v_rm_ts >= start_time) & (self.exp.v_rm_ts < end_time)
            stroke_voxels = self.v_rm_colors[stroke_mask]
        
        if len(stroke_voxels) == 0:
            return {"Green": 0, "Yellow": 0, "Red": 0, "Other": 100}
            
        # Normalize alpha values and classify
        alpha_values = stroke_voxels[:, 3] / 255.0
        labels = [self.classify_voxel_color(round(alpha, 1)) for alpha in alpha_values]
        
        # Calculate percentages
        total_voxels = len(labels)
        color_counts = pd.Series(labels).value_counts(normalize=True) * 100
        
        return {
            "Green": color_counts.get("Green", 0),
            "Yellow": color_counts.get("Yellow", 0),
            "Red": color_counts.get("Red", 0),
            "Other": color_counts.get("Other", 0)
        }