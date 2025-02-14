import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)


import numpy as np
from scipy import ndimage
from src.unet_vda_model import VelocityDealiaser
from src.utils import compute_coordinates


def closest_index(array, target):
    """
    Finds the index of the value in the array closest to the target value.
    
    Parameters:
        array (numpy.ndarray): The array to search.
        target (float): The target value.
    
    Returns:
        int: The index of the closest value.
    """
    # Compute the absolute difference between each element and the target
    diff = np.abs(array - target)
    
    # Find the index of the minimum difference
    index = np.argmin(diff)
    
    return index

def calculate_range_folded_mask(radar_data, nyquist_velocity, velocity_data=None, 
                              signal_to_noise=None, snr_threshold=10):
    """
    Calculate range folding mask for radar data.
    
    Parameters:
    -----------
    radar_data : numpy.ndarray
        Base radar reflectivity or velocity data
    nyquist_velocity : float
        Nyquist velocity of the radar in m/s
    velocity_data : numpy.ndarray, optional
        Doppler velocity data if different from radar_data
    signal_to_noise : numpy.ndarray, optional
        Signal-to-noise ratio data
    snr_threshold : float, optional
        Threshold for SNR to consider valid data (default=10 dB)
        
    Returns:
    --------
    mask : numpy.ndarray
        Boolean mask where True indicates range folded data
    """
    # Initialize mask
    mask = np.zeros_like(radar_data, dtype=bool)
    
    # If velocity data is provided, use it for folding detection
    if velocity_data is not None:
        # Look for velocity values near Nyquist limits
        nyquist_margin = 0.00 # 0% margin
        mask |= (np.abs(velocity_data) > (1 - nyquist_margin) * nyquist_velocity)
    
    # Check for sudden changes in reflectivity or velocity
    gradient = np.gradient(radar_data, axis=1)
    threshold = np.std(gradient) * 3  # 3 sigma threshold
    mask |= (np.abs(gradient) > threshold)
    
    # If SNR data is available, use it to refine the mask
    if signal_to_noise is not None:
        mask |= (signal_to_noise < snr_threshold)
    
    # Apply spatial continuity check
    # Range folding typically affects continuous regions
    structure = np.ones((3, 3))
    mask = ndimage.binary_closing(mask, structure=structure)
    
    return mask

def nearest_neighbor_interpolation(input_array, target_shape):
    """
    Interpolates a 2D array along the first axis using nearest-neighbor method.
    
    Args:
        input_array (np.ndarray): The input array to interpolate. 
                                  Should have shape (n, m).
        target_shape (tuple): The desired shape (new_n, m), where new_n > n.
    
    Returns:
        np.ndarray: The interpolated array with shape (new_n, m).
    """
    n, m = input_array.shape
    new_n, _ = target_shape
    assert new_n == 2 * n, "Target shape must be double the first dimension of the input array"
    
    # Initialize an empty array for the output
    output_array = np.zeros((new_n, m), dtype=input_array.dtype)
    
    # Perform nearest-neighbor interpolation
    for i in range(n):
        output_array[2 * i] = input_array[i]       # Copy original row to even index
        output_array[2 * i + 1] = input_array[i]   # Copy original row to odd index
    
    return output_array

def wrap_slice(data, start_idx, length, axis=0):
    """
    Creates a slice of the data with wrapping around the edges if needed.
    
    Args:
        data: numpy array to slice
        start_idx: starting index
        length: desired length of slice
        axis: axis along which to slice (default 0)
    
    Returns:
        Wrapped slice of the data array
    """
    data_length = data.shape[axis]
    
    if start_idx + length <= data_length:
        # Normal case - no wrapping needed
        return np.take(data, range(start_idx, start_idx + length), axis=axis)
    else:
        # Wrapping needed
        first_part = np.take(data, range(start_idx, data_length), axis=axis)
        remaining_length = length - (data_length - start_idx)
        second_part = np.take(data, range(remaining_length), axis=axis)
        return np.concatenate([first_part, second_part], axis=axis)


class RadarDataProcessor:
    def __init__(self, num_range=1152):
        self.velocity_dealiaser = VelocityDealiaser(
            model_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model_checkpoint/unet_vda.onnx")
        )
        self.num_range = num_range

    def process_kdp_data(self, kdp_radar_data):
        kdp_data = np.ma.filled(kdp_radar_data.fields['specific_differential_phase']['data'], np.nan)
        kdp_unsorted_az_angles = kdp_radar_data.get_azimuth(0)
        sorted_indices = np.argsort(kdp_unsorted_az_angles)
        kdp_data = kdp_data[sorted_indices, :]
        # kdp_data = nearest_neighbor_interpolation(kdp_data, (kdp_data.shape[1]*2, kdp_data.shape[0]))
        kdp_range_values = kdp_radar_data.range['data']
        azimuth_kdp_values = np.sort(kdp_unsorted_az_angles)
        return {
            'kdp_data': kdp_data,
            'kdp_range_values': kdp_range_values,
            'kdp_azimuth_values': azimuth_kdp_values
        }
    
    def process_lv2_data(self, lv2_radar_data):
        s0 = lv2_radar_data.extract_sweeps(np.array([0]))
        s0_azm = s0.get_azimuth(0)
        s0_azm_as = np.argsort(s0_azm)

        s1 = lv2_radar_data.extract_sweeps(np.array([1]))
        s1_azm = s1.get_azimuth(0)
        s1_azm_as = np.argsort(s1_azm)

        s2 = lv2_radar_data.extract_sweeps(np.array([2]))
        s2_azm = s2.get_azimuth(0)
        s2_azm_as = np.argsort(s2_azm)

        s3 = lv2_radar_data.extract_sweeps(np.array([3]))
        s3_azm = s3.get_azimuth(0)
        s3_azm_as = np.argsort(s3_azm)

        variables_1 = {
            'REF': 'reflectivity',
            'RHO': 'cross_correlation_ratio',
            'ZDR': 'differential_reflectivity'
        }

        variables_2 = {
            'REF_2': 'reflectivity',
            'VEL': 'velocity',
            'SW': 'spectrum_width'
        }
            
        nyquist_5 = s1.get_nyquist_vel(0, check_uniform=False)
        nyquist_9 = s3.get_nyquist_vel(0, check_uniform=False)

        tilt_5 = {}
        tilt_9 = {}

        tilt_5['nyquist_velocity'] = nyquist_5
        tilt_9['nyquist_velocity'] = nyquist_9

        for var in variables_1.keys():
            tilt_5[var] = np.ma.filled(s0.fields[variables_1[var]]['data'][s0_azm_as],np.nan)
            tilt_9[var] = np.ma.filled(s2.fields[variables_1[var]]['data'][s2_azm_as],np.nan)

        for var in variables_2.keys():
            tilt_5[var] = np.ma.filled(s1.fields[variables_2[var]]['data'][s1_azm_as],np.nan)
            tilt_9[var] = np.ma.filled(s3.fields[variables_2[var]]['data'][s3_azm_as],np.nan)

        range_values = s0.range['data']
        azimuth_values = np.sort(s0_azm)
        return {
            'tilt_5': tilt_5,
            'tilt_9': tilt_9,
            'range_values': range_values,
            'azimuth_values': azimuth_values
        }
    
    def preprocess(self, lv2_radar_data, kdp_radar_data_tilt05, kdp_radar_data_tilt09):
        # Process KDP data for both tilts
        kdp_tilt05_data = self.process_kdp_data(kdp_radar_data_tilt05)
        kdp_tilt09_data = self.process_kdp_data(kdp_radar_data_tilt09)
        
        # Extract KDP data and coordinates
        kdp_data_tilt05 = kdp_tilt05_data['kdp_data']
        kdp_tilt05_range_values = kdp_tilt05_data['kdp_range_values'] 
        kdp_tilt05_azimuth_values = kdp_tilt05_data['kdp_azimuth_values']
        
        kdp_data_tilt09 = kdp_tilt09_data['kdp_data']
        kdp_tilt09_range_values = kdp_tilt09_data['kdp_range_values']
        kdp_tilt09_azimuth_values = kdp_tilt09_data['kdp_azimuth_values']

        # Process Level 2 radar data
        lv2_data = self.process_lv2_data(lv2_radar_data)
        lv2_data_tilt05 = lv2_data['tilt_5']
        lv2_data_tilt09 = lv2_data['tilt_9'] 
        lv2_range_values = lv2_data['range_values']
        lv2_azimuth_values = lv2_data['azimuth_values']

        # Dealias velocity data
        tilt_5_vel_original = lv2_data_tilt05['VEL']
        tilt_9_vel_original = lv2_data_tilt09['VEL']

        nyquist_5 = lv2_data_tilt05['nyquist_velocity']
        nyquist_9 = lv2_data_tilt09['nyquist_velocity']

        dealiased_vel_tilt05 = self.velocity_dealiaser.dealias(np.expand_dims(tilt_5_vel_original[:, :1152], axis=(0, -1)), np.expand_dims(nyquist_5, axis=0))
        dealiased_vel_tilt09 = self.velocity_dealiaser.dealias(np.expand_dims(tilt_9_vel_original[:, :1152], axis=(0, -1)), np.expand_dims(nyquist_9, axis=0))

        lv2_data_tilt05['VEL'] = np.concatenate((dealiased_vel_tilt05[:, :1152], tilt_5_vel_original[:, 1152:]), axis=1)
        lv2_data_tilt09['VEL'] = np.concatenate((dealiased_vel_tilt09[:, :1152], tilt_9_vel_original[:, 1152:]), axis=1)

        lv2_data_tilt05['range_folded_mask'] = calculate_range_folded_mask(
            radar_data=lv2_data_tilt05['REF_2'],
            nyquist_velocity=nyquist_5,
            velocity_data=lv2_data_tilt05['VEL']
        ).astype(int)

        lv2_data_tilt09['range_folded_mask'] = calculate_range_folded_mask(
            radar_data=lv2_data_tilt09['REF_2'],
            nyquist_velocity=nyquist_9,
            velocity_data=lv2_data_tilt09['VEL']
        ).astype(int)

        kdp_data_tilt05 = nearest_neighbor_interpolation(kdp_data_tilt05, (kdp_data_tilt05.shape[0]*2, kdp_data_tilt05.shape[1]))
        kdp_data_tilt09 = nearest_neighbor_interpolation(kdp_data_tilt09, (kdp_data_tilt09.shape[0]*2, kdp_data_tilt09.shape[1]))

        lv2_data_tilt05['KDP'] = kdp_data_tilt05
        lv2_data_tilt09['KDP'] = kdp_data_tilt09

        for var in lv2_data_tilt05:
            lv2_data_tilt05[var] = np.expand_dims(lv2_data_tilt05[var], axis=(0, -1))

        for var in lv2_data_tilt09:
            lv2_data_tilt09[var] = np.expand_dims(lv2_data_tilt09[var], axis=(0, -1))


        # if min_azimuth < 0:
        #     new_min_azimuth = min_azimuth + lv2_azimuth_values[-1]
        #     new_min_azimuth_kdp_05 = min_azimuth + kdp_tilt05_azimuth_values[-1]
        #     new_min_azimuth_kdp_09 = min_azimuth + kdp_tilt09_azimuth_values[-1]
        # else:
        #     new_min_azimuth = min_azimuth
        #     new_min_azimuth_kdp_05 = min_azimuth
        #     new_min_azimuth_kdp_09 = min_azimuth

        # if min_range < 0:
        #     new_min_range = min_range + lv2_range_values[-1]
        #     new_min_range_kdp_05 = min_range + kdp_tilt05_range_values[-1]
        #     new_min_range_kdp_09 = min_range + kdp_tilt09_range_values[-1]
        # else:
        #     new_min_range = min_range
        #     new_min_range_kdp_05 = min_range
        #     new_min_range_kdp_09 = min_range

        # rng_idx = closest_index(lv2_range_values, new_min_range)
        # rng_kdp_05_idx = closest_index(kdp_tilt05_range_values, new_min_range_kdp_05)
        # rng_kdp_09_idx = closest_index(kdp_tilt09_range_values, new_min_range_kdp_09)

        # azm_idx = closest_index(lv2_azimuth_values, new_min_azimuth)  
        # azm_kdp_05_idx = closest_index(kdp_tilt05_azimuth_values, new_min_azimuth_kdp_05)  
        # azm_kdp_09_idx = closest_index(kdp_tilt09_azimuth_values, new_min_azimuth_kdp_09)  

        # for var in lv2_data_tilt05:
        #     if var == "KDP":
        #         # First handle azimuth wrapping (axis 0)
        #         azm_slice = wrap_slice(lv2_data_tilt05[var], azm_idx, 120, axis=0)
        #         # Then handle range wrapping (axis 1)
        #         full_slice = wrap_slice(azm_slice, rng_kdp_05_idx, 240, axis=1)
        #         # Add the required dimensions
        #         lv2_data_tilt05[var] = np.expand_dims(full_slice, axis=(0, -1))
        #     else:
        #         # First handle azimuth wrapping (axis 0)
        #         azm_slice = wrap_slice(lv2_data_tilt05[var], azm_idx, 120, axis=0)
        #         # Then handle range wrapping (axis 1)
        #         full_slice = wrap_slice(azm_slice, rng_idx, 240, axis=1)
        #         # Add the required dimensions
        #         lv2_data_tilt05[var] = np.expand_dims(full_slice, axis=(0, -1))
        
        # for var in lv2_data_tilt09:
        #     if var == "KDP":
        #         azm_slice = wrap_slice(lv2_data_tilt09[var], azm_kdp_09_idx, 120, axis=0)
        #         full_slice = wrap_slice(azm_slice, rng_kdp_09_idx, 240, axis=1)
        #         lv2_data_tilt09[var] = np.expand_dims(full_slice, axis=(0, -1))
        #     else:
        #         azm_slice = wrap_slice(lv2_data_tilt09[var], azm_kdp_09_idx, 120, axis=0)
        #         full_slice = wrap_slice(azm_slice, rng_idx, 240, axis=1)
        #         lv2_data_tilt09[var] = np.expand_dims(full_slice, axis=(0, -1))

        lv2_data_tilt05['DBZ'] = lv2_data_tilt05['REF']
        lv2_data_tilt05['WIDTH'] = lv2_data_tilt05['SW']
        lv2_data_tilt05['RHOHV'] = lv2_data_tilt05['RHO']
        lv2_data_tilt09['DBZ'] = lv2_data_tilt09['REF']
        lv2_data_tilt09['WIDTH'] = lv2_data_tilt09['SW']
        lv2_data_tilt09['RHOHV'] = lv2_data_tilt09['RHO']
        # lv2_data_tilt05['event_id'] = np.array([0])
        # lv2_data_tilt05['ef_number'] = np.array([0])
        # lv2_data_tilt05['time'] = np.array([100])
        lv2_data_tilt05['az_lower'] = np.array([lv2_azimuth_values[0]])
        lv2_data_tilt05['az_upper'] = np.array([lv2_azimuth_values[-1]])
        lv2_data_tilt05['rng_lower'] = np.array([lv2_range_values[0]])
        lv2_data_tilt05['rng_upper'] = np.array([lv2_range_values[-1]])

        lv2_data_tilt09['az_lower'] = np.array([lv2_azimuth_values[0]])
        lv2_data_tilt09['az_upper'] = np.array([lv2_azimuth_values[-1]])
        lv2_data_tilt09['rng_lower'] = np.array([lv2_range_values[0]])
        lv2_data_tilt09['rng_upper'] = np.array([lv2_range_values[-1]])

        variables = [
            'DBZ', 'VEL', 'KDP', 'RHOHV', 'ZDR', 'WIDTH',
            'range_folded_mask'
        ]

        meta_data = [
            # 'event_id', 'ef_number', 'time', 
            'az_lower', 'az_upper', 'rng_lower', 'rng_upper', 'nyquist_velocity'
        ]


        preprocessed_data = {}
        for var in variables:
            level2_tilt05_var = lv2_data_tilt05[var]
            level2_tilt09_var = lv2_data_tilt09[var]
            if level2_tilt05_var.shape[2] < self.num_range:
                level2_tilt05_var = np.pad(
                    level2_tilt05_var,
                    ((0, 0), (0, 0), (0, self.num_range - level2_tilt05_var.shape[2]), (0, 0)),
                    mode='constant',
                    constant_values=np.nan
                )
            else:
                level2_tilt05_var = level2_tilt05_var[:, :, :self.num_range, :]
            if level2_tilt09_var.shape[2] < self.num_range:
                level2_tilt09_var = np.pad(
                    level2_tilt09_var,
                    ((0, 0), (0, 0), (0, self.num_range - level2_tilt09_var.shape[2]), (0, 0)),
                    mode='constant',
                    constant_values=np.nan
                )
            else:
                level2_tilt09_var = level2_tilt09_var[:, :, :self.num_range, :]
            preprocessed_data[var] = np.concatenate((level2_tilt05_var, level2_tilt09_var), axis=-1)

        for var in meta_data:
            preprocessed_data[var] = lv2_data_tilt05[var]
        

        preprocessed_data['coordinates'] = np.expand_dims(
            compute_coordinates(
                preprocessed_data,
                tilt_last=True,
                include_az=False,
                backend=np
            ),
            axis=0
        )

        # print(preprocessed_data['nyquist_velocity'])
        # print(preprocessed_data['az_lower'])
        # print(preprocessed_data['az_upper'])
        # print(preprocessed_data['rng_lower'])
        # print(preprocessed_data['rng_upper'])


        return preprocessed_data
    
if __name__ == "__main__":
    from src.data_downloader import RadarDataDownloader
    radar_downloader = RadarDataDownloader(radar_station='KTLX', dest_folder='data/lv2_data')
    radar_processor = RadarDataProcessor(num_range=1152)
    lv2_radar_file = radar_downloader.download_nextrad_2_file(date="20250213", time="000611")
    if os.path.exists(lv2_radar_file):
        lv2_radar_data = radar_downloader.read_nextrad_2_file(lv2_radar_file)
        os.remove(lv2_radar_file)
    kdp_radar_data_tilt05 = radar_downloader.get_nexrad_l3_data(product='N0K', date='20250213', time='000611')
    kdp_radar_data_tilt09 = radar_downloader.get_nexrad_l3_data(product='N3K', date='20250213', time='000611')
    preprocessed_data = radar_processor.preprocess(lv2_radar_data, kdp_radar_data_tilt05, kdp_radar_data_tilt09)
    print(preprocessed_data.keys())
