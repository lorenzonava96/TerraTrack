import numpy as np
import matplotlib.pyplot as plt
import rasterio
from scipy.ndimage import zoom
from scipy.ndimage import gaussian_filter, binary_fill_holes, convolve, zoom
from collections import defaultdict
from sklearn.cluster import DBSCAN
from rasterio.transform import from_bounds
import cv2
from skimage.morphology import thin, binary_closing, binary_dilation

def resample_morpho_to_match(orig_shape, morpho_path, output_path):
    """
    Resample a three-band morpho image (DEM, slope, aspect) to match the shape of the displacement field (orig).
    """
    with rasterio.open(morpho_path) as src:
        dem = src.read(1)  # Band 1: DEM
        slope = src.read(2)  # Band 2: Slope
        aspect = src.read(3)  # Band 3: Aspect
        transform = src.transform

    # Calculate zoom factors for height and width
    zoom_y = orig_shape[0] / dem.shape[0]
    zoom_x = orig_shape[1] / dem.shape[1]

    # Resample DEM, slope, and aspect using zoom
    resampled_dem = zoom(dem, (zoom_y, zoom_x), order=1)  # Linear interpolation for DEM
    resampled_slope = zoom(slope, (zoom_y, zoom_x), order=1)  # Linear interpolation for slope
    resampled_aspect = zoom(aspect, (zoom_y, zoom_x), order=1)  # Linear interpolation for aspect

    # Calculate new transform
    scale_x = transform[0] / zoom_x  # Horizontal resolution
    scale_y = -transform[4] / zoom_y  # Vertical resolution (negative for correct orientation)
    new_transform = rasterio.transform.from_origin(
        transform.c, transform.f, scale_x, scale_y
    )

    # Save the resampled three-band morpho image
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=resampled_dem.shape[0],
        width=resampled_dem.shape[1],
        count=3,  # Three bands: DEM, slope, aspect
        dtype=dem.dtype,
        crs=src.crs,
        transform=new_transform,
    ) as dst:
        dst.write(resampled_dem.astype(dem.dtype), 1)  # Write DEM to band 1
        dst.write(resampled_slope.astype(slope.dtype), 2)  # Write slope to band 2
        dst.write(resampled_aspect.astype(aspect.dtype), 3)  # Write aspect to band 3

    return output_path, new_transform

def accumulate_displacement(all_u, all_v, all_feature_points, separation):
    """
    Accumulate displacements for each feature point across all pairs.
    """
    displacement_data = defaultdict(lambda: {'u_values': [], 'v_values': [], 'years_diff': []})
    for i in range(len(all_u)):
        u, v, feature_points = all_u[i], all_v[i], all_feature_points[i]
        years_diff = separation[i]
        for j in range(len(u)):
            x, y = feature_points[j]
            displacement_data[(x, y)]['u_values'].append(u[j])
            displacement_data[(x, y)]['v_values'].append(v[j])
            displacement_data[(x, y)]['years_diff'].append(years_diff)
    return displacement_data

def calculate_median_displacement(displacement_data, pixel_size, min_displacement_threshold, max_displacement_threshold):
    """
    Calculate median displacement vectors, their magnitude, and angle.

    Parameters:
    - displacement_data: Dictionary with accumulated displacements.
    - pixel_size: Scaling factor for displacements.
    - min_displacement_threshold: Minimum magnitude threshold for filtering.
    - max_displacement_threshold: Maximum magnitude threshold for filtering.

    Returns:
    - median_feature_points: Array of feature points.
    - median_u: Array of median U displacements.
    - median_v: Array of median V displacements.
    - median_magnitude: Array of displacement magnitudes.
    - median_angles: Array of displacement angles (radians).
    """
    import numpy as np

    median_feature_points, median_u, median_v, median_magnitude, median_angles = [], [], [], [], []

    for (x, y), data in displacement_data.items():
        if len(data['u_values']) > 0 and len(data['years_diff']) > 0:
            # Normalize each displacement by years_diff
            normalized_u = np.array(data['u_values']) / np.array(data['years_diff'])
            normalized_v = np.array(data['v_values']) / np.array(data['years_diff'])

            # Compute the median normalized velocities
            median_u_velocity = np.nanmedian(normalized_u)
            median_v_velocity = np.nanmedian(normalized_v)

            # Scale velocities to displacements using pixel_size
            median_u_disp = median_u_velocity * pixel_size
            median_v_disp = median_v_velocity * pixel_size

            # Compute magnitude and angle
            magnitude = np.sqrt(median_u_disp**2 + median_v_disp**2)
            angle = np.arctan2(median_v_disp, median_u_disp)

            # Apply magnitude filtering
            if min_displacement_threshold <= magnitude <= max_displacement_threshold:
                median_feature_points.append((x, y))
                median_u.append(median_u_disp)
                median_v.append(median_v_disp)
                median_magnitude.append(magnitude)
                median_angles.append(angle)

    return (
        np.array(median_feature_points),
        np.array(median_u),
        np.array(median_v),
        np.array(median_magnitude),
        np.array(median_angles),
    )

def filter_angular_coherence(median_angles, angular_threshold, smoothing_sigma=1):
    """
    Filter vectors based on angular coherence.
    """
    # Ensure angles are wrapped between -pi and pi
    median_angles = (median_angles + np.pi) % (2 * np.pi) - np.pi

    # Convert angles to complex exponential form
    exp_angles = np.exp(1j * median_angles)

    # Apply Gaussian smoothing on real and imaginary parts
    smoothed_real = gaussian_filter(exp_angles.real, sigma=smoothing_sigma)
    smoothed_imag = gaussian_filter(exp_angles.imag, sigma=smoothing_sigma)

    # Reconstruct smoothed angles
    smoothed_angles = np.angle(smoothed_real + 1j * smoothed_imag)

    # Compute angular deviation
    angular_deviation = np.abs((median_angles - smoothed_angles + np.pi) % (2 * np.pi) - np.pi)

    # Return valid indices where angular deviation is within the threshold
    return angular_deviation <= angular_threshold

def calculate_slope(dem_path):
    """
    Calculate slope from a DEM.
    """
    with rasterio.open(dem_path) as src:
        dem = src.read(1)
        transform = src.transform

    # Extract pixel sizes from the transform
    pixel_size_x = transform[0]  # Pixel width
    pixel_size_y = abs(transform[4])  # Pixel height (absolute value to avoid negative spacing)

    # Calculate gradients (dz/dx and dz/dy)
    x_grad, y_grad = np.gradient(dem, pixel_size_x, pixel_size_y)

    # Calculate slope in radians, then convert to degrees
    slope = np.arctan(np.sqrt(x_grad**2 + y_grad**2)) * (180 / np.pi)

    return slope, transform

def filter_slope(median_feature_points, resampled_slope, min_slope_threshold):
    """
    Filter vectors based on slope values directly using indices.
    """
    # Ensure points are valid indices for the slope array
    cols = np.round(median_feature_points[:, 0]).astype(int)
    rows = np.round(median_feature_points[:, 1]).astype(int)

    # Mask to ensure indices are within bounds
    valid_mask = (
        (rows >= 0) & (rows < resampled_slope.shape[0]) &
        (cols >= 0) & (cols < resampled_slope.shape[1])
    )

    # Apply valid mask
    rows = rows[valid_mask]
    cols = cols[valid_mask]

    # Retrieve slope values at valid points
    slope_values = resampled_slope[rows, cols]

    # Filter based on slope threshold
    slope_mask = slope_values >= min_slope_threshold

    # Combine valid and slope masks
    combined_mask = np.zeros(len(median_feature_points), dtype=bool)
    combined_mask[np.where(valid_mask)[0]] = slope_mask

    # Return indices of valid points meeting the slope threshold
    return combined_mask

def filter_aspect(median_feature_points, u, v, resampled_aspect, aspect_tolerance=45):
    """
    Filter vectors based on their alignment with the downslope direction derived from aspect.

    Parameters:
    - median_feature_points: Array of feature point coordinates (x, y).
    - u: Horizontal displacements.
    - v: Vertical displacements.
    - resampled_aspect: 2D array of aspect values (in degrees, 0°=east, CCW).
    - aspect_tolerance: Angular tolerance (in degrees) for vector alignment.

    Returns:
    - combined_mask: Boolean array indicating valid points aligned with the downslope direction.
    """
    # Ensure points are valid indices for the aspect array
    cols = np.round(median_feature_points[:, 0]).astype(int)
    rows = np.round(median_feature_points[:, 1]).astype(int)

    # Mask to ensure indices are within bounds
    valid_mask = (
        (rows >= 0) & (rows < resampled_aspect.shape[0]) &
        (cols >= 0) & (cols < resampled_aspect.shape[1])
    )

    # Apply valid mask
    rows = rows[valid_mask]
    cols = cols[valid_mask]
    u_valid = u[valid_mask]
    v_valid = v[valid_mask]

    # Retrieve aspect values at valid points
    aspect_values = resampled_aspect[rows, cols]

    # Convert aspect to north-origin clockwise convention
    # Aspect (0°=east, CCW) -> Aspect (0°=north, CW)
    aspect_values_converted = (-aspect_values - 270) % 360

    # Calculate the vector direction in degrees
    vector_angles = np.degrees(np.arctan2(-v_valid, u_valid))  # Negative v for image coordinates
    vector_angles = (vector_angles + 360) % 360  # Normalize to [0, 360]

    # Calculate the angular difference between vector direction and aspect
    angular_diff = np.abs(vector_angles - aspect_values_converted)
    angular_diff = np.minimum(angular_diff, 360 - angular_diff)  # Wrap around 360 degrees

    # Check if the angular difference is within the tolerance
    alignment_mask = angular_diff <= aspect_tolerance

    # Combine valid and alignment masks
    combined_mask = np.zeros(len(median_feature_points), dtype=bool)
    combined_mask[np.where(valid_mask)[0]] = alignment_mask

    return combined_mask

def filter_clusters(median_feature_points, median_u, median_v, median_magnitude, eps=10, min_samples=5):
    """
    Remove isolated displacements using DBSCAN clustering.
    """
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(median_feature_points)
    labels = clustering.labels_
    clustered_indices = labels >= 0
    return (
        median_feature_points[clustered_indices],
        median_u[clustered_indices],
        median_v[clustered_indices],
        median_magnitude[clustered_indices]
    )

def plot_displacement_field(median_feature_points, median_u, median_v, median_magnitude, orig, arrow_scale):
    """
    Plot the filtered and smoothed displacement field.
    """
    plt.figure(figsize=(16, 16))
    plt.imshow(orig, cmap='gray')

    quiver = plt.quiver(
        median_feature_points[:, 0], median_feature_points[:, 1],
        median_u, median_v, median_magnitude,
        angles='xy', scale_units='xy', scale=arrow_scale, cmap='jet', width=0.001
    )

    plt.colorbar(quiver, label="Displacement Rate (m/year)", fraction=0.02)
    plt.title("Median Displacement Field (in m/year)")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.show()

def filter_final_map(
    median_feature_points, median_u, median_v, median_magnitude, median_angles,
    resampled_slope, resampled_aspect, study_area_image, 
    angular_threshold_degrees=30, min_slope_threshold=5, 
    aspect_tolerance=45, smoothing_sigma=1, clustering_params=(10, 5),
    use_angular_coherence=True, use_slope_filter=True, 
    use_aspect_filter=True, use_clustering=True, arrow_scale=1,
    manual_threshold=None
):
    """
    Filters the median displacement field with optional filtering steps.
    
    Before any filtering, displacements are thresholded by magnitude. 
    If `manual_threshold` is provided (i.e., not None), it is used as the threshold;
    otherwise, the threshold is set to the 95th percentile of `median_magnitude`.
    
    Args:
        median_feature_points (array): Feature point coordinates.
        median_u (array): U-displacement components.
        median_v (array): V-displacement components.
        median_magnitude (array): Displacement magnitudes.
        median_angles (array): Displacement angles (in degrees).
        resampled_slope (array): Slope data (in degrees) for the study area.
        resampled_aspect (array): Aspect data (in degrees) for the study area.
        study_area_image (object): Image of the study area (for visualization).
        angular_threshold_degrees (float): Angular threshold in degrees.
        min_slope_threshold (float): Minimum slope in degrees.
        aspect_tolerance (float): Allowable deviation from downslope direction in degrees.
        smoothing_sigma (int): Gaussian smoothing scale.
        clustering_params (tuple): DBSCAN parameters (eps, min_samples).
        use_angular_coherence (bool): Apply angular coherence filter.
        use_slope_filter (bool): Apply slope filter.
        use_aspect_filter (bool): Apply aspect filter.
        use_clustering (bool): Apply clustering filter.
        arrow_scale (float): Scale for displacement arrow plotting.
        manual_threshold (float or None): The manual threshold value to use. If None, the
            95th percentile of `median_magnitude` is used.
    
    Returns:
        median_feature_points, median_u, median_v, median_magnitude:
            The filtered median displacement field.
    """
    # Convert angular threshold to radians
    angular_threshold = np.deg2rad(angular_threshold_degrees)

    # Step 1: Threshold by displacement magnitude
    if manual_threshold is None:
        threshold_value = np.percentile(median_magnitude, 95)
        print('Using 95th Percentile of the Median Map as threshold:', threshold_value)
    else:
        threshold_value = manual_threshold
        print('Using manual threshold value:', threshold_value)
    
    valid_indices = median_magnitude > threshold_value
    median_feature_points = median_feature_points[valid_indices]
    median_u = median_u[valid_indices]
    median_v = median_v[valid_indices]
    median_magnitude = median_magnitude[valid_indices]
    median_angles = median_angles[valid_indices]

    if median_feature_points.shape[0] <= 1:
        print("Insufficient valid points after thresholding.")
        return median_feature_points, median_u, median_v, median_magnitude

    # Step 2: Optional - Filter by Angular Coherence
    if use_angular_coherence and median_feature_points.shape[0] > 1:
        valid_indices = filter_angular_coherence(median_angles, angular_threshold)
        median_feature_points = median_feature_points[valid_indices]
        median_u = median_u[valid_indices]
        median_v = median_v[valid_indices]
        median_magnitude = median_magnitude[valid_indices]
        median_angles = median_angles[valid_indices]

    if median_feature_points.shape[0] <= 1:
        print("Insufficient valid points after angular coherence filtering. Skipping further filters.")
        return median_feature_points, median_u, median_v, median_magnitude

    # Step 3: Optional - Filter by Slope
    if use_slope_filter:
        valid_indices = filter_slope(median_feature_points, resampled_slope, min_slope_threshold)
        median_feature_points = median_feature_points[valid_indices]
        median_u = median_u[valid_indices]
        median_v = median_v[valid_indices]
        median_magnitude = median_magnitude[valid_indices]

    if median_feature_points.shape[0] <= 1:
        print("Insufficient valid points after slope filtering. Skipping further filters.")
        return median_feature_points, median_u, median_v, median_magnitude

    # Step 4: Optional - Filter Non-Downslope Movements
    if use_aspect_filter:
        valid_indices = filter_aspect(median_feature_points, median_u, median_v, resampled_aspect, aspect_tolerance)
        median_feature_points = median_feature_points[valid_indices]
        median_u = median_u[valid_indices]
        median_v = median_v[valid_indices]
        median_magnitude = median_magnitude[valid_indices]

    if median_feature_points.shape[0] <= 1:
        print("Insufficient valid points after aspect filtering. Skipping further filters.")
        return median_feature_points, median_u, median_v, median_magnitude

    # Step 5: Optional - Filter by Clustering
    if use_clustering:
        eps, min_samples = clustering_params
        median_feature_points, median_u, median_v, median_magnitude = filter_clusters(
            median_feature_points, median_u, median_v, median_magnitude, eps=eps, min_samples=min_samples
        )

    if median_feature_points.shape[0] <= 1:
        print("Insufficient valid points after clustering. Skipping further filters.")
        return median_feature_points, median_u, median_v, median_magnitude

    # Step 6: Plot the final filtered median displacement field (optional)
    print("Plotting Median Displacement Field...")
    plot_displacement_field(median_feature_points, median_u, median_v, median_magnitude, study_area_image, arrow_scale)

    return median_feature_points, median_u, median_v, median_magnitude

def calculate_median_displacement(displacement_data, pixel_size):
    """
    Calculate median displacement vectors, their magnitude, and angle.

    Parameters:
    - displacement_data: Dictionary with accumulated displacements.
    - pixel_size: Scaling factor for displacements.
    - min_displacement_threshold: Minimum magnitude threshold for filtering.
    - max_displacement_threshold: Maximum magnitude threshold for filtering.

    Returns:
    - median_feature_points: Array of feature points.
    - median_u: Array of median U displacements.
    - median_v: Array of median V displacements.
    - median_magnitude: Array of displacement magnitudes.
    - median_angles: Array of displacement angles (radians).
    """

    median_feature_points, median_u, median_v, median_magnitude, median_angles = [], [], [], [], []

    for (x, y), data in displacement_data.items():
        if len(data['u_values']) > 0 and len(data['years_diff']) > 0:
            # Normalize each displacement by years_diff
            normalized_u = np.array(data['u_values']) / np.array(data['years_diff'])
            normalized_v = np.array(data['v_values']) / np.array(data['years_diff'])

            # Compute the median normalized velocities
            median_u_velocity = np.nanmedian(normalized_u)
            median_v_velocity = np.nanmedian(normalized_v)

            # Scale velocities to displacements using pixel_size
            median_u_disp = median_u_velocity * pixel_size
            median_v_disp = median_v_velocity * pixel_size

            # Compute magnitude and angle
            magnitude = np.sqrt(median_u_disp**2 + median_v_disp**2)
            angle = np.arctan2(median_v_disp, median_u_disp)

            # Apply magnitude filtering
            # if min_displacement_threshold <= magnitude <= max_displacement_threshold:
            median_feature_points.append((x, y))
            median_u.append(median_u_disp)
            median_v.append(median_v_disp)
            median_magnitude.append(magnitude)
            median_angles.append(angle)

    return (
        np.array(median_feature_points),
        np.array(median_u),
        np.array(median_v),
        np.array(median_magnitude),
        np.array(median_angles),
    )

def create_magnitude_map(image_shape, feature_points, magnitudes, block_size, overlap):
    """
    Maps magnitudes onto a grid based on feature points, block size, and overlap.
    
    Parameters:
        image_shape (tuple): Shape of the image (height, width).
        feature_points (array): Array of (x, y) coordinates for feature magnitudes.
        magnitudes (array): Array of magnitude values corresponding to feature points.
        block_size (int): Size of each block (assumes square blocks).
        overlap (float): Overlap fraction between blocks (0.0 to 1.0).
    
    Returns:
        np.ndarray: 2D array (grid) with magnitudes mapped to corresponding blocks.
    """
    # Calculate step size based on block size and overlap
    step_size = int(block_size * (1 - overlap))
    
    # Determine grid dimensions
    grid_height = (image_shape[0] - block_size) // step_size + 1
    grid_width = (image_shape[1] - block_size) // step_size + 1
    
    # Initialize the magnitude map grid
    magnitude_map = np.zeros((grid_height, grid_width), dtype=np.float32)

    for (x, y), magnitude in zip(feature_points, magnitudes):
        # Map pixel coordinates to grid coordinates
        grid_x = int((x - (block_size / 2)) // step_size)
        grid_y = int((y     - (block_size / 2)) // step_size)

        
        # Ensure grid indices are within bounds
        if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
            magnitude_map[grid_y, grid_x] += magnitude
    
    return magnitude_map

def overlay_magnitude_map(study_area_image, magnitude_map, block_size, overlap):
    """
    Overlay the magnitude map onto the study area image.
    
    Parameters:
        study_area_image (np.ndarray): The original image (background).
        magnitude_map (np.ndarray): The lower-resolution magnitude map.
        block_size (int): Block size used for the magnitude map.
        overlap (float): Overlap percentage used for the magnitude map.
    """
    # Calculate scale factors for upsampling
    scale_factor_y = study_area_image.shape[0] / magnitude_map.shape[0]
    scale_factor_x = study_area_image.shape[1] / magnitude_map.shape[1]
    
    # Upscale the magnitude map to match the study area image size
    upscaled_magnitude_map = zoom(magnitude_map, (scale_factor_y, scale_factor_x), order=0)
    
    # Plot the study area with the overlaid magnitude map
    plt.figure(figsize=(18, 10))
    plt.imshow(study_area_image, cmap='gray', interpolation='nearest')  # Grayscale background
    plt.imshow(upscaled_magnitude_map, cmap='Reds', alpha=0.5, interpolation='nearest')  # Transparent red overlay
    plt.colorbar(label='Displacement Rate (m/year)')
    plt.title('Motion Map')
    plt.xlabel('Width (pixels)')
    plt.ylabel('Height (pixels)')
    plt.show()

def create_raster_maps(feature_points, u_values, v_values, study_area_image, block_size, overlap):
    """
    Create raster maps for u, v, magnitude, and angle from feature tracking data.
    """

    # Calculate step size based on block size and overlap
    step_size = int(block_size * (1 - overlap))

    # Determine grid dimensions
    grid_height = (study_area_image.shape[0] - block_size) // step_size + 1
    grid_width  = (study_area_image.shape[1] - block_size) // step_size + 1
    out_map_shape = (grid_height, grid_width)

    # Initialize raster maps with NaN (to track missing data)
    u_map = np.full(out_map_shape, np.nan, dtype=np.float32)
    v_map = np.full(out_map_shape, np.nan, dtype=np.float32)
    magnitude_map = np.full(out_map_shape, np.nan, dtype=np.float32)
    angle_map = np.full(out_map_shape, np.nan, dtype=np.float32)

    for ((x, y), u_val, v_val) in zip(feature_points, u_values, v_values):
        # Map pixel coordinates to grid
        grid_x = int((x - (block_size / 2)) // step_size)
        grid_y = int((y - (block_size / 2)) // step_size)

        # Ensure indices are within bounds
        if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
            u_map[grid_y, grid_x] = u_val
            v_map[grid_y, grid_x] = v_val

            # Compute magnitude and angle
            mag, ang = cv2.cartToPolar(np.array([u_val]), np.array([v_val]), angleInDegrees=True)
            magnitude_map[grid_y, grid_x] = mag[0]
            angle_map[grid_y, grid_x] = ang[0]

    # Replace NaNs with interpolated values
    u_map = np.nan_to_num(u_map, nan=0)
    v_map = np.nan_to_num(v_map, nan=0)
    magnitude_map = np.nan_to_num(magnitude_map, nan=0)
    angle_map = np.nan_to_num(angle_map, nan=0)

    return u_map, v_map, magnitude_map, angle_map

def plot_maps(u_map, v_map, magnitude_map, angle_map):
    """
    Plot the u, v, magnitude, and angle maps.
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    im0 = axs[0, 0].imshow(u_map, cmap='viridis')
    axs[0, 0].set_title('U Map')
    plt.colorbar(im0, ax=axs[0, 0])

    im1 = axs[0, 1].imshow(v_map, cmap='viridis')
    axs[0, 1].set_title('V Map')
    plt.colorbar(im1, ax=axs[0, 1])

    im2 = axs[1, 0].imshow(magnitude_map, cmap='viridis')
    axs[1, 0].set_title('Magnitude Map')
    plt.colorbar(im2, ax=axs[1, 0])

    im3 = axs[1, 1].imshow(angle_map, cmap='hsv')
    axs[1, 1].set_title('Angle Map (Degrees)')
    plt.colorbar(im3, ax=axs[1, 1])

    plt.tight_layout()
    plt.show()

def process_mask(mask):
    """
    Apply morphological operations to a binary mask:
    'bridge', 'thicken', 'majority', 'fill', and 'thin'.
    
    Parameters:
        mask (np.ndarray): Binary mask.
    
    Returns:
        np.ndarray: Processed binary mask.
    """
    # Ensure mask is boolean
    mask = mask.astype(bool)
    
    # 'bridge': Connect pixels using binary closing with a 3x3 footprint.
    mask = binary_closing(mask, footprint=np.ones((3, 3)))
    #plt.figure(figsize=(6,6))
    #plt.imshow(mask, cmap='gray')
    #plt.title("Mask after 'bridge' (binary closing)")
    #plt.colorbar()
    #plt.tight_layout()
    #plt.show()
    
    # 'thicken' by one iteration: simulate by binary dilation.
    mask = binary_dilation(mask)
    #plt.figure(figsize=(6,6))
    #plt.imshow(mask, cmap='gray')
    #plt.title("Mask after first 'thicken' (binary dilation)")
    #plt.colorbar()
    #plt.tight_layout()
    #plt.show()
    
    # Apply 'bridge' again.
    mask = binary_closing(mask, footprint=np.ones((3, 3)))
    #plt.figure(figsize=(6,6))
    #plt.imshow(mask, cmap='gray')
    #plt.title("Mask after second 'bridge' (binary closing)")
    #plt.colorbar()
    #plt.tight_layout()
    #plt.show()
    
    # 'thicken' again.
    mask = binary_dilation(mask)
    #plt.figure(figsize=(6,6))
    #plt.imshow(mask, cmap='gray')
    #plt.title("Mask after second 'thicken' (binary dilation)")
    #plt.colorbar()
    #plt.tight_layout()
    #plt.show()
    
    # 'majority': Set pixel to True if 5 or more pixels in its 3x3 neighborhood are True.
    kernel = np.ones((3, 3))
    neighbor_count = convolve(mask.astype(np.uint8), kernel, mode='constant', cval=0)
    mask = neighbor_count >= 5
    #plt.figure(figsize=(6,6))
    #plt.imshow(mask, cmap='gray')
    #plt.title("Mask after 'majority' filtering")
    #plt.colorbar()
    #plt.tight_layout()
    #plt.show()
    
    # 'fill': Fill holes in the binary mask.
    mask = binary_fill_holes(mask)
    #plt.figure(figsize=(6,6))
    #plt.imshow(mask, cmap='gray')
    #plt.title("Mask after 'fill' (hole filling)")
    #plt.colorbar()
    #plt.tight_layout()
    #plt.show()
    
    # 'thin' by one iteration.
    mask = thin(mask, max_num_iter=1)
    #plt.figure(figsize=(6,6))
    #plt.imshow(mask, cmap='gray')
    #plt.colorbar()
    #plt.title("Mask after 'thin'")
    #plt.tight_layout()
    #plt.show()
    
    return mask

def save_as_geotiff(orig_path, output_path, magnitude_map, block_size, overlap):
    """
    Save the magnitude map as a georeferenced GeoTIFF.
    
    Parameters:
        orig_path (str): Path to the original georeferenced image.
        output_path (str): Path to save the output GeoTIFF file.
        magnitude_map (np.ndarray): The magnitude map to save.
        block_size (int): Block size used in magnitude map creation.
        overlap (float): Overlap fraction used for the magnitude map.
    """
    # Open the original georeferenced image
    with rasterio.open(orig_path) as src:
        # Extract bounding box and CRS
        bounds = src.bounds
        crs = src.crs
        
        # Calculate the resolution of the magnitude map
        step_size = int(block_size * (1 - overlap))
        grid_height, grid_width = magnitude_map.shape
        
        # Compute the affine transform for the new grid resolution
        transform = from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top,
                                grid_width, grid_height)
    
    # Write the magnitude map as a GeoTIFF file
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=grid_height,
        width=grid_width,
        count=1,
        dtype=magnitude_map.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(magnitude_map, 1)
