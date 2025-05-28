# classes/gen_metrics.py
import numpy as np
import pandas as pd
from collections import defaultdict
from utils.anatomy_dict import anatomy_dict
from utils.hdf5_utils import save_dict_to_hdf5

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
        Returns a dictionary with the number of voxels removed (value) for each anatomy / region (key).
        """
        # Returns a dictionary with the number of voxels removed (value) for each anatomy (key)
        vxl_rmvd_dict = defaultdict(int)
        v_rm_colors = np.array(self.exp.v_rm_colors).astype(np.int32)
        v_rm_colors_df = pd.DataFrame(v_rm_colors, columns=["ts_idx", "r", "g", "b", "a"])

        # Initialize columns
        v_rm_colors_df["anatomy_name"] = None
        v_rm_colors_df["region_name"] = None
    
        # add a column with the anatomy names
        for name, anatomy_info_list in anatomy_dict.items():
            color_str, full_name = anatomy_info_list
            color_parts = color_str.split(" ")
            # Handle RGB-based anatomy
            if len(color_parts) == 3:
                r, g, b = map(int, color_parts)
                mask = (
                    (v_rm_colors_df["r"] == r) &
                    (v_rm_colors_df["g"] == g) &
                    (v_rm_colors_df["b"] == b)
                )
                v_rm_colors_df.loc[mask, "anatomy_name"] = name
            # Handle region-based (alpha-encoded) anatomy
            elif len(color_parts) == 1:
                alpha_threshold = float(color_parts[0])
                alpha_int = round(alpha_threshold * 255)
                mask = (v_rm_colors_df["a"] == alpha_int)
                v_rm_colors_df.loc[mask, "region_name"] = name
        
        # Count voxels by anatomy
        anatomy_counts = v_rm_colors_df["anatomy_name"].value_counts(dropna=True)
        for anatomy, count in anatomy_counts.items():
            vxl_rmvd_dict[anatomy] += count

        # Count voxels by region
        region_counts = v_rm_colors_df["region_name"].value_counts(dropna=True)
        for region, count in region_counts.items():
            vxl_rmvd_dict[region] += count
        
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