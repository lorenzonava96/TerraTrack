from skimage.registration import phase_cross_correlation
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import zoom, median_filter, convolve, median_filter, gaussian_filter, center_of_mass
from scipy.fft import fft2, ifft2, fftshift
from scipy.ndimage import gaussian_filter
from numpy.lib.stride_tricks import sliding_window_view
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import os

# =============================================================================
# Displacement Analysis Function
# =============================================================================
def displacement_analysis(
    img1, img2,
    method='block_matching',  # Options: 'optical_flow', 'block_matching'
    block_size=16,
    overlap=0.8,
    match_func='fft_ncc',  # For block matching: 'phase_cross_corr', 'fft_ncc', 'median_dense_optical_flow', etc.
    subpixel_method="parabolic",  # Only used if match_func is 'fft_ncc'
    zero_mask=None,  # Optional mask for invalid areas
    filter_params=None,  # Dictionary of parameters for filter_displacements
    plot=True,  # Whether to visualize the displacement field
    arrow_scale=10
):
    """
    Compute subpixel displacements between two images using either dense optical flow 
    or block matching. For block matching, multiple matching functions are available.
    
    Parameters:
    -----------
    img1 : ndarray
        First image (reference).
    img2 : ndarray
        Second image (target).
    method : str
        Overall method to use. Options:
         - 'optical_flow': Uses dense optical flow (with built-in subpixel refinement).
         - 'block_matching': Uses block matching.
    block_size : int
        Size of the blocks to compare (used only in block matching).
    overlap : float
        Overlap ratio between blocks (0 to 1, used only in block matching).
    match_func : str or callable, optional
        For block matching, select the matching function. Options include:
         - 'phase_cross_corr': Uses phase cross-correlation (subpixel refinement is embedded).
         - 'fft_ncc': Uses FFT-based normalized cross-correlation with batch processing;
                      allows selection of subpixel refinement via the `subpixel_method` parameter.
         - 'median_dense_optical_flow': Uses dense optical flow within each block.
         - Or a custom function.
        If not provided, a default (e.g., 'phase_cross_corr') is used.
    subpixel_method : str
        For custom NCC ('fft_ncc') block matching, choose subpixel refinement method.
        Options: 'center_of_mass', 'quadratic', 'parabolic'. (Ignored for other match_func choices.)
    zero_mask : ndarray, optional
        Mask for invalid regions.
    filter_params : dict, optional
        Dictionary of parameters for displacement filtering (passed to filter_displacements).
    plot : bool
        Whether to plot the displacement field.
    arrow_scale : float or int
        Scale for arrows in the displacement plot.
        
    Returns:
    --------
    u : ndarray
        Array of horizontal displacements.
    v : ndarray
        Array of vertical displacements.
    feature_points : ndarray
        Coordinates (x, y) of feature points.
    pkrs : ndarray
        Peak-to-peak ratios (if available).
    snrs : ndarray
        Signal-to-noise ratios (if available).
        
    Notes:
    ------
    - When using optical_flow or default matching functions (e.g., phase_cross_corr or 
      median_dense_optical_flow), the subpixel refinement is embedded in the function and the
      `subpixel_method` parameter is ignored.
    - For the custom NCC ('fft_ncc'), the subpixel refinement strategy is applied as selected,
      and batch processing is used for acceleration.
    - This unified API allows future expansion to include options such as GPU-accelerated 
      implementations (e.g., 'gpu_ncc_parabolic') or faster NCC variants.
    """
    if filter_params is None:
        filter_params = {}

    # Default match_func for block matching if none provided:
    if method == 'block_matching' and match_func is None:
        match_func = 'phase_cross_corr'

    u, v, feature_points, pkrs, snrs = None, None, None, None, None

    # Step 1: Compute displacements using the selected method.
    if method == 'optical_flow':
        feature_points, u, v, pkrs, snrs = dense_optical_flow_displacement(img1, img2)
    elif method == 'block_matching':
        feature_points, u, v, pkrs, snrs = block_matching_vectorized(
            img1, img2,
            block_size=block_size,
            overlap=overlap,
            match_func=match_func,
            subpixel_method=subpixel_method  # Only used if match_func == 'fft_ncc'
        )
    else:
        raise ValueError("Invalid method. Options are 'optical_flow' or 'block_matching'.")

    # Step 2: Filter displacements.
    u, v, feature_points = filter_displacements(
        u, v, feature_points, zero_mask,
        pkr_values=pkrs, snr_values=snrs,
        **filter_params
    )

    # Step 3: Optionally plot the displacement field.
    if plot and feature_points.size > 0:
        plot_displacement_field(u, v, feature_points, img1, arrow_scale=arrow_scale)

    return u, v, feature_points, pkrs, snrs

def dense_optical_flow_displacement(img1, img2):
    """
    Compute feature points, displacements (u, v), and errors using dense optical flow.

    Optimized for faster computation by tuning Farneback parameters and using efficient numpy operations.

    Parameters:
    - img1: First image (grayscale).
    - img2: Second image (grayscale).

    Returns:
    - feature_points: Array of [x, y] coordinates for each pixel.
    - u: Array of horizontal displacements (dx).
    - v: Array of vertical displacements (dy).
    - errors: None (placeholder for future error estimation).
    """
    # Adjust Farneback parameters for speed optimization
    flow = cv2.calcOpticalFlowFarneback(
        img1, img2, None,
        pyr_scale=0.5,  # Increased scale for faster pyramid building
        levels=5,       # Reduced levels for less computation
        winsize=15,     # Larger window for better accuracy with fewer iterations
        iterations=3,   # Reduced iterations for faster convergence
        poly_n=5,       # Smaller neighborhood size for faster computation
        poly_sigma=1.2, # Adjusted smoothing for polynomial expansion
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN  # Use Gaussian window for faster and smoother flow estimation
    )

    # Extract horizontal and vertical flow components
    u = flow[..., 0]
    v = flow[..., 1]

    # Create feature points grid more efficiently
    h, w = img1.shape
    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
    feature_points = np.column_stack((x_coords.ravel(), y_coords.ravel()))

    # Flatten displacement fields
    u = u.ravel()
    v = v.ravel()

    return feature_points, u, v, None, None

import numpy as np
import cv2
from scipy.fftpack import fft2, ifft2, fftshift
from scipy.ndimage import center_of_mass
from numpy.lib.stride_tricks import sliding_window_view
from skimage.registration import phase_cross_correlation

# ------------------------------
# Helper Function for Subpixel Refinement
# ------------------------------

def parabolic_interpolation(values, peak_index):
    """Parabolic interpolation for subpixel refinement."""
    if peak_index <= 0 or peak_index >= len(values) - 1:
        return peak_index  # No interpolation at edges
    left = values[peak_index - 1]
    center = values[peak_index]
    right = values[peak_index + 1]
    return peak_index + 0.5 * (left - right) / (left - 2 * center + right)

# ------------------------------
# Batch FFT-NCC Matching Function
# ------------------------------
def batch_fft_ncc(blocks1, blocks2, subpixel_method="parabolic"):
    """
    Process a batch of blocks using FFT-based normalized cross-correlation.
    
    Parameters:
      blocks1 : ndarray of shape (N, block_size, block_size)
      blocks2 : ndarray of shape (N, block_size, block_size)
      subpixel_method : str, one of "center_of_mass", "quadratic", "parabolic"
      
    Returns:
      dx, dy : ndarrays (N,) of subpixel displacements (the refined offsets)
      pkr, snr : ndarrays (N,) of the peak-to-peak ratios and signal-to-noise ratios
    """
    N, bs, _ = blocks1.shape
    # Compute FFT for all blocks (vectorized over N)
    FFT1 = fft2(blocks1)
    FFT2 = fft2(blocks2)
    cross_power = FFT1 * np.conj(FFT2)
    cross_corr = ifft2(cross_power / (np.abs(cross_power) + 1e-10)).real
    # Apply fftshift along last two axes for each block
    cross_corr = np.fft.fftshift(cross_corr, axes=(1, 2))
    
    # For each block, find the index of the maximum value
    flattened = cross_corr.reshape(N, -1)
    peak_flat_idx = np.argmax(flattened, axis=1)
    i_indices = peak_flat_idx // bs
    j_indices = peak_flat_idx % bs
    max_corr = cross_corr[np.arange(N), i_indices, j_indices]
    mean_corr = np.mean(np.abs(cross_corr), axis=(1,2))
    snr = max_corr / (mean_corr + 1e-10)
    
    # For each block, determine the second highest peak (for PKR) and perform subpixel refinement
    pkr = np.zeros(N)
    dx = np.zeros(N)
    dy = np.zeros(N)
    buffer_size = 1
    for k in range(N):
        block = cross_corr[k]
        i = i_indices[k]
        j = j_indices[k]
        # Mask out a small region around the primary peak
        temp = block.copy()
        i_min = max(0, i - buffer_size)
        i_max = min(bs, i + buffer_size + 1)
        j_min = max(0, j - buffer_size)
        j_max = min(bs, j + buffer_size + 1)
        temp[i_min:i_max, j_min:j_max] = -np.inf
        second_peak = np.max(temp)
        pkr[k] = max_corr[k] / (second_peak + 1e-10)
        
        # Subpixel refinement per block
        if subpixel_method == "center_of_mass":
            window_size = 3
            x_min = max(0, j - window_size // 2)
            x_max = min(bs, j + window_size // 2 + 1)
            y_min = max(0, i - window_size // 2)
            y_max = min(bs, i + window_size // 2 + 1)
            local_window = block[y_min:y_max, x_min:x_max]
            refined_y, refined_x = center_of_mass(local_window)
            # Convert from local window to block-centered coordinates:
            dx[k] = refined_x + x_min - bs // 2
            dy[k] = refined_y + y_min - bs // 2
        elif subpixel_method == "parabolic":
            dx[k] = parabolic_interpolation(block[i, :], j)
            dy[k] = parabolic_interpolation(block[:, j], i)
        else:
            raise ValueError("Invalid subpixel_method. Choose from 'center_of_mass', 'quadratic', or 'parabolic'.")
    
    return dx, dy, pkr, snr

# ------------------------------
# Loop-based Matching Functions
# ------------------------------
def phase_cross_corr_vec(block1, block2):
    """Matching using phase cross-correlation (non-batchable)."""
    shifts, error, _ = phase_cross_correlation(block1, block2, upsample_factor=10)
    return shifts[1], shifts[0], None, None

def median_dense_optical_flow(block1, block2):
    """Matching using dense optical flow (non-batchable)."""
    flow = cv2.calcOpticalFlowFarneback(
        block1, block2, None, 
        pyr_scale=0.5,
        levels=3,
        winsize=block1.shape[0],
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )
    dx = -np.median(flow[..., 0])
    dy = -np.median(flow[..., 1])
    return dx, dy, None, None

# ------------------------------
# Main Block Matching Function (with batch branch)
# ------------------------------
def block_matching_vectorized(img1, img2, block_size=16, overlap=0.8,
                              match_func='phase_cross_corr',
                              subpixel_method="center_of_mass"):
    """
    Perform vectorized block matching between two images.

    Parameters:
      img1: First image (reference).
      img2: Second image (target).
      block_size: Size of the blocks to compare.
      overlap: Overlap percentage between blocks (0 to 1).
      match_func: Matching function to use. Options:
                  'phase_cross_corr', 'fft_ncc', 'median_dense_optical_flow',
                  or a custom callable.
      subpixel_method: Subpixel refinement method ('center_of_mass', 'quadratic', 'parabolic')
                       (only used if match_func is 'fft_ncc').

    Returns:
      feature_points: Array of [x, y] coordinates of matched blocks.
      u: Array of horizontal displacements.
      v: Array of vertical displacements.
      pkrs: Array of peak-to-peak ratios.
      snrs: Array of signal-to-noise ratios.
    """
    # Calculate stride based on block size and overlap
    stride = int(block_size * (1 - overlap))
    img_shape = img1.shape

    # Create sliding window views for blocks
    blocks1 = sliding_window_view(img1, (block_size, block_size))[::stride, ::stride]
    blocks2 = sliding_window_view(img2, (block_size, block_size))[::stride, ::stride]
    num_blocks = blocks1.shape[0] * blocks1.shape[1]
    blocks1 = blocks1.reshape(num_blocks, block_size, block_size)
    blocks2 = blocks2.reshape(num_blocks, block_size, block_size)

    # Decide processing strategy based on match_func:
    if match_func == 'fft_ncc':
        # Use batch processing for fft_ncc
        dx, dy, pkrs, snrs = batch_fft_ncc(blocks1, blocks2, subpixel_method=subpixel_method)
        # Invert displacements as in your convention
        u = -dx
        v = -dy
    else:
        # Map available matching functions to their implementations.
        match_func_map = {
            'phase_cross_corr': phase_cross_corr_vec,
            'median_dense_optical_flow': median_dense_optical_flow,
        }
        if callable(match_func):
            match_func_callable = match_func
        elif match_func in match_func_map:
            match_func_callable = match_func_map[match_func]
        else:
            raise ValueError(f"Invalid match_func: {match_func}. Options: {list(match_func_map.keys())} or a callable.")

        # Initialize result arrays
        u = np.zeros(num_blocks)
        v = np.zeros(num_blocks)
        pkrs = np.zeros(num_blocks)
        snrs = np.zeros(num_blocks)
        # Process each block in a loop
        for idx in range(num_blocks):
            dx, dy, pkr, snr = match_func_callable(blocks1[idx], blocks2[idx])
            u[idx] = -dx
            v[idx] = -dy
            pkrs[idx] = pkr
            snrs[idx] = snr

    # Generate feature point coordinates
    grid_x, grid_y = np.meshgrid(
        np.arange(0, img_shape[1] - block_size + 1, stride) + block_size // 2,
        np.arange(0, img_shape[0] - block_size + 1, stride) + block_size // 2,
    )
    feature_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    return feature_points, u, v, pkrs, snrs


def filter_displacements(
    u, 
    v, 
    feature_points, 
    zero_mask,
    apply_magnitude_filter=False, 
    min_magnitude=0.1, 
    max_magnitude=10,
    apply_zero_mask_filter=False,
    apply_deviation_filter=False, 
    std_factor=2,
    apply_remove_median_displacement=False,
    apply_median_filter_step=False, 
    filter_size=3,
    apply_angular_coherence_filter=False, 
    angular_threshold=30, 
    smoothing_sigma=1,
    apply_erratic_displacement_filter=False, 
    neighborhood_size=5, 
    deviation_threshold=2,
    pkr_values=None,
    apply_pkr_filter=False,
    pkr_threshold=1.3,
    snr_values=None,
    apply_snr_filter=False,
    snr_threshold=3
):
    """
    Filter displacement vectors and optionally remove erratic displacements.

    Args:
        u, v: Displacement components (arrays of the same shape).
        feature_points: Corresponding feature points for the displacements (N x 2).
        zero_mask: Mask for invalid regions (1 for invalid, 0 for valid).
        apply_magnitude_filter: Whether to filter by displacement magnitude.
        min_magnitude: Minimum valid displacement magnitude.
        max_magnitude: Maximum valid displacement magnitude.
        apply_zero_mask_filter: Whether to apply the zero mask filter.
        apply_deviation_filter: Whether to filter by deviation from mean displacement.
        std_factor: Threshold for deviation filtering based on standard deviation.
        apply_remove_median_displacement: Whether to remove the overall median displacement.
        apply_median_filter_step: Whether to apply median filtering to the valid displacements.
        filter_size: Size of the median filter for smoothing.
        apply_angular_coherence_filter: Whether to apply angular coherence filtering.
        angular_threshold: Threshold in degrees for angular deviation.
        smoothing_sigma: Standard deviation for Gaussian smoothing of angles.
        apply_erratic_displacement_filter: Whether to remove erratic displacements based on neighborhood stats.
        neighborhood_size: Size of the spatial neighborhood for the median filter.
        deviation_threshold: Threshold for deviation from neighborhood median/mean magnitude.
        snr_values: Array of SNR values associated with each displacement (same length as u, v).
        apply_snr_filter: Whether to filter by an SNR threshold.
        snr_threshold: Minimum SNR for a displacement to be considered valid.

    Returns:
        (u_filtered, v_filtered, filtered_feature_points): 
            The filtered displacement components and their corresponding feature points.
    """

    def filter_angular_coherence(angles, angular_threshold_deg, sigma=1):
        """
        Filter vectors based on angular coherence. Smooth angles using a complex exponential
        representation and Gaussian filtering, then compare the deviation from the smoothed angles.
        """
        # Normalize angles to [-pi, pi]
        angles = (angles + np.pi) % (2 * np.pi) - np.pi

        # Exponential map of angles
        exp_angles = np.exp(1j * angles)

        # Smooth real and imaginary parts
        smoothed_real = gaussian_filter(exp_angles.real, sigma=sigma)
        smoothed_imag = gaussian_filter(exp_angles.imag, sigma=sigma)

        # Get smoothed angles
        smoothed_angles = np.angle(smoothed_real + 1j * smoothed_imag)

        # Compute angular deviation and convert threshold to radians
        angular_deviation = np.abs((angles - smoothed_angles + np.pi) % (2 * np.pi) - np.pi)
        angular_threshold_rad = np.deg2rad(angular_threshold_deg)

        return angular_deviation <= angular_threshold_rad

    # Step 1: Filter by magnitude
    if apply_magnitude_filter:
        magnitudes = np.sqrt(u**2 + v**2)
        valid_mask = (magnitudes > min_magnitude) & (magnitudes < max_magnitude)
    else:
        valid_mask = np.ones_like(u, dtype=bool)

    # Step 2: Apply the zero mask
    if apply_zero_mask_filter:
        # Coordinates in feature_points are [x, y], so be sure to index correctly
        # or flip if your feature_points is in (row, col) or (x, y) format
        x_coords = np.clip(feature_points[:, 1].astype(int), 0, zero_mask.shape[0] - 1)
        y_coords = np.clip(feature_points[:, 0].astype(int), 0, zero_mask.shape[1] - 1)

        # zero_mask == 0 means valid area in your description
        mask_exclusion = (zero_mask[x_coords, y_coords] == 0)
        valid_mask &= mask_exclusion

    # Step 3: Apply deviation filter
    if apply_deviation_filter and np.any(valid_mask):
        mean_u = np.mean(u[valid_mask])
        mean_v = np.mean(v[valid_mask])
        std_u = np.std(u[valid_mask])
        std_v = np.std(v[valid_mask])

        deviation_mask = (
            (np.abs(u - mean_u) < std_factor * std_u) &
            (np.abs(v - mean_v) < std_factor * std_v)
        )
        valid_mask &= deviation_mask

    # Step 4: Apply angular coherence filter
    if apply_angular_coherence_filter and np.any(valid_mask):
        angles = np.arctan2(v, u)
        angular_mask = filter_angular_coherence(angles, angular_threshold, smoothing_sigma)
        valid_mask &= angular_mask

    # Step 5: Remove overall median displacement
    if apply_remove_median_displacement and np.any(valid_mask):
        median_u = np.median(u[valid_mask])
        median_v = np.median(v[valid_mask])
        u = u - median_u
        v = v - median_v

    # Step 6: Remove erratic displacements
    if apply_erratic_displacement_filter and np.any(valid_mask):
        u_neighborhood = median_filter(u, size=neighborhood_size)
        v_neighborhood = median_filter(v, size=neighborhood_size)

        neighborhood_magnitudes = np.sqrt(u_neighborhood**2 + v_neighborhood**2)
        current_magnitudes = np.sqrt(u**2 + v**2)

        erratic_mask = np.abs(current_magnitudes - neighborhood_magnitudes) < deviation_threshold
        valid_mask &= erratic_mask

    # Step 7: Apply median filter to valid displacements
    # We only filter the valid entries. Then we replace the original ones with filtered values.
    if apply_median_filter_step and np.any(valid_mask):
        # Grab arrays of valid u, v only
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]

        # Apply median filter (1D) on these valid values
        u_valid_filtered = median_filter(u_valid, size=filter_size)
        v_valid_filtered = median_filter(v_valid, size=filter_size)

        # Put them back
        u[valid_mask] = u_valid_filtered
        v[valid_mask] = v_valid_filtered

    # Step 8: Apply PKR filter (only if we have at least one valid PKR)
    if apply_pkr_filter and pkr_values is not None:
        # If the entire array is NaN, skip the PKR filter:
        if np.isnan(pkr_values).all():
            pass
            # print("PKR array is all NaN. Skipping PKR filter...")
        else:
            # Apply your PKR filtering logic here
            pkr_mask = pkr_values >= pkr_threshold
            valid_mask &= pkr_mask

    # Step 9: Apply SNR filter (only if we have at least one valid SNR)
    if apply_snr_filter and snr_values is not None:
        # If the entire array is NaN, skip the SNR filter:
        if np.isnan(snr_values).all():
            pass
            # print("SNR array is all NaN. Skipping SNR filter...")
        else:
            # Apply your SNR filtering logic here
            snr_mask = snr_values >= snr_threshold
            valid_mask &= snr_mask

    # Handle empty results
    if not np.any(valid_mask):
        print("No valid displacements after filtering.")
        return np.array([]), np.array([]), np.empty((0, 2))

    # Return filtered results
    u_filtered = u[valid_mask]
    v_filtered = v[valid_mask]
    filtered_feature_points = feature_points[valid_mask]

    return u_filtered, v_filtered, filtered_feature_points

def plot_displacement_field(u, v, feature_points, img1, num_arrows=10, arrow_scale=10):
    plt.figure(figsize=(12, 12))
    plt.imshow(img1, cmap='gray')
    magnitudes = np.sqrt(u ** 2 + v ** 2)
    quiver = plt.quiver(
        feature_points[:, 0], feature_points[:, 1], u, v, magnitudes,
        angles='xy', scale_units='xy', scale=arrow_scale, width=0.0025, headwidth=3, cmap='jet'
    )
    cbar = plt.colorbar(quiver, label='Displacement Magnitude (pixels/frame)', fraction=0.026, pad=0.04)
    cbar.ax.tick_params(labelsize=10)
    plt.title('Displacement Field at Feature Points', fontsize=16)
    plt.xlabel('X Coordinate', fontsize=12)
    plt.ylabel('Y Coordinate', fontsize=12)
    plt.show()


from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time

def process_image_pairs(dat1, dat2, datax, preprocessed_stack, filter_params, zero_mask=None,
                        method='block_matching', block_size=32, overlap=0.8, 
                        match_func='phase_cross_corr', subpixel_method='center_of_mass', 
                        max_workers=None, parallel=True):
    """
    Processes image pairs to compute displacements, either in parallel or sequentially,
    printing progress at intervals.

    Parameters:
    - dat1, dat2: Lists of start and end dates for each pair.
    - datax: List of all available dates corresponding to image indices.
    - preprocessed_stack: 3D array of preprocessed images (H x W x Time).
    - filter_params: Dictionary of filtering parameters for displacement analysis.
    - zero_mask: Optional mask for invalid regions.
    - method: Method to use ('block_matching' or 'optical_flow').
    - block_size: Block size for block matching.
    - overlap: Overlap percentage for block matching.
    - match_func: Matching function for block matching (e.g., 'phase_cross_corr', 'fft_ncc', 'median_dense_optical_flow').
    - subpixel_method: Subpixel refinement method (used only if match_func is 'fft_ncc').
    - max_workers: Maximum number of threads to use (default: os.cpu_count()).
    - parallel: Boolean flag to enable parallel processing (default: True).

    Returns:
    - all_results: List of results with displacements and feature points for each pair.
    """
    # Map dates to indices
    date_to_index = {date: idx for idx, date in enumerate(datax)}

    # Helper function to process a single pair
    def process_pair(i):
        try:
            date1, date2 = dat1[i], dat2[i]
            if date1 not in date_to_index or date2 not in date_to_index:
                return None  # Skip invalid dates

            idx1, idx2 = date_to_index[date1], date_to_index[date2]
            if idx1 >= preprocessed_stack.shape[2] or idx2 >= preprocessed_stack.shape[2]:
                return None  # Skip out-of-bounds indices

            # Load images for the valid date pair
            img1, img2 = preprocessed_stack[:, :, idx1], preprocessed_stack[:, :, idx2]

            # Perform displacement analysis (subpixel_method is passed here)
            u, v, feature_points, pkrs, snrs = displacement_analysis(
                img1=img1,
                img2=img2,
                method=method,
                block_size=block_size,
                overlap=overlap,
                match_func=match_func,
                subpixel_method=subpixel_method,
                zero_mask=zero_mask,
                filter_params=filter_params,
                plot=False
            )

            # Return results if valid displacements are found
            if len(u) > 0 and len(v) > 0:
                return {
                    "u": u,
                    "v": v,
                    "feature_points": feature_points,
                    "pkrs": pkrs,
                    "snrs": snrs,
                    "img1": img1
                }
            else:
                return None
        except Exception as e:
            print(f"Error processing pair {i}: {e}")
            return None

    total_pairs = len(dat1)
    all_results = []
    processed_count = 0

    print(f"Processing {total_pairs} image pairs...")
    start_time = time.time()

    if parallel:
        with ThreadPoolExecutor(max_workers=max_workers or os.cpu_count()) as executor:
            futures = {executor.submit(process_pair, i): i for i in range(total_pairs)}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    all_results.append(result)
                processed_count += 1
                if processed_count % 10 == 0 or processed_count == total_pairs:
                    print(f"Processed {processed_count}/{total_pairs} pairs...")
    else:
        for i in range(total_pairs):
            result = process_pair(i)
            if result:
                all_results.append(result)
            processed_count += 1
            if processed_count % 10 == 0 or processed_count == total_pairs:
                print(f"Processed {processed_count}/{total_pairs} pairs...")

    elapsed_time = time.time() - start_time
    print(f"\nProcessed {total_pairs} pairs in {elapsed_time:.2f} seconds.")

    return all_results
