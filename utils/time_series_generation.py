import numpy as np
import pandas as pd
import rasterio
import random
import string
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial

def accumulate_displacement_with_placeholders(all_u, all_v, all_feature_points, separation,
                                                median_feature_points, dat1, dat2, pixel_size,
                                                all_pkrs, all_snrs):
    """
    Accumulate displacements for each filtered feature point (median_feature_points) across all pairs,
    inserting NaN for missing intervals. Displacements are converted to meters using pixel size.
    Velocities are also computed for each interval, and additional parameters all_pkrs and all_snrs
    are saved along with the rest.
    """
    
    # Ensure points are consistently tuples
    median_feature_points = [tuple(fp) for fp in median_feature_points]

    # Initialize the data structure for filtered points only, including keys for pkrs and snrs.
    displacement_data = {tuple(fp): {
        'u_values': np.full(len(dat1), np.nan),
        'v_values': np.full(len(dat1), np.nan),
        'years_diff': np.full(len(dat1), np.nan),
        'dat1': [None] * len(dat1),
        'dat2': [None] * len(dat1),
        'u_velocity': np.full(len(dat1), np.nan),
        'v_velocity': np.full(len(dat1), np.nan),
        'pkrs': np.full(len(dat1), np.nan),
        'snrs': np.full(len(dat1), np.nan),
    } for fp in median_feature_points}

    # Debugging: Ensure all input lists have matching lengths
    if not (len(all_u) == len(all_v) == len(all_feature_points) == len(separation) ==
            len(dat1) == len(dat2) == len(all_pkrs) == len(all_snrs)):
        raise ValueError("Input lists must have the same length")

    # Loop over all intervals
    for i in range(len(all_u)):
        u, v, feature_points, year_diff = all_u[i], all_v[i], all_feature_points[i], separation[i]

        # Convert feature points to tuples for consistency
        feature_points = [tuple(fp) for fp in feature_points]

        for j, point in enumerate(feature_points):
            if point in displacement_data:  # Only consider filtered points
                # Convert displacements to meters using pixel size
                u_meters = u[j] * pixel_size
                v_meters = v[j] * pixel_size

                displacement_data[point]['u_values'][i] = -u_meters
                displacement_data[point]['v_values'][i] = -v_meters
                displacement_data[point]['years_diff'][i] = year_diff
                displacement_data[point]['dat1'][i] = dat1[i]
                displacement_data[point]['dat2'][i] = dat2[i]
                displacement_data[point]['pkrs'][i] = all_pkrs[i][j]  # Use index j to get the scalar value.
                displacement_data[point]['snrs'][i] = all_snrs[i][j]  # Use index j here as well.

                # Calculate velocities (m/year) if year_diff is valid
                if not np.isnan(year_diff) and year_diff > 0:
                    u_velocity = u_meters / year_diff
                    v_velocity = v_meters / year_diff
                    displacement_data[point]['u_velocity'][i] = u_velocity
                    displacement_data[point]['v_velocity'][i] = v_velocity
                    
    return displacement_data

def compute_rmse_polynomial(velocities, timestamps, degree=3):
    """
    Compute RMSE for a velocity time series after fitting a polynomial trend.

    Parameters:
    - velocities: List or array of velocity values.
    - timestamps: List or array of timestamps corresponding to the velocity values.
    - degree: Degree of the polynomial to fit (default is 3 for a cubic polynomial).

    Returns:
    - RMSE value.
    """
    velocities = np.array(velocities)
    timestamps = np.array(timestamps)

    # Filter out NaN values from velocities and their corresponding timestamps
    valid_indices = ~np.isnan(velocities)
    velocities = velocities[valid_indices]
    timestamps = timestamps[valid_indices]

    if len(velocities) == 0:
        raise ValueError("No valid data points available for RMSE computation.")

    # Normalize timestamps to improve numerical stability
    t_norm = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min())

    # Fit a polynomial of the specified degree to the data
    coefs = Polynomial.fit(t_norm, velocities, degree).convert().coef

    # Compute the fitted values (predicted velocities)
    fitted_values = np.polyval(coefs[::-1], t_norm)  # Reverse coefficients for np.polyval

    # Calculate residuals
    residuals = velocities - fitted_values

    # Calculate RMSE
    rmse = np.sqrt(np.mean(residuals**2))
    return rmse

def create_velocity_time_series(displacement_data):
    """
    Create velocity time series for each point by assigning velocities to the midpoint date
    between dat1 and dat2.

    Parameters:
    - displacement_data: Dictionary with 'dat1', 'dat2', 'u_velocities', and 'v_velocities' for each point.
    - interval: Aggregation interval ('daily', 'monthly', 'yearly').

    Returns:
    - midpoint_dfs: Dictionary of midpoint DataFrames for each point.
    """
    midpoint_dfs = {}

    for point, data in displacement_data.items():
        # Initialize an empty list to store midpoint velocities
        midpoint_data = []

        # Extract data for the point
        dat1 = data['dat1']
        dat2 = data['dat2']
        u_velocities = data['u_velocity']
        v_velocities = data['v_velocity']

        # Populate midpoint velocities for each interval
        for start, end, u_vel, v_vel in zip(dat1, dat2, u_velocities, v_velocities):
            if not np.isnan(u_vel) and not np.isnan(v_vel):
                # Calculate the midpoint date
                start_date = pd.Timestamp(start)
                end_date = pd.Timestamp(end)
                midpoint_date = start_date + (end_date - start_date) / 2

                # Append the midpoint velocity data
                midpoint_data.append({'date': midpoint_date, 'u_velocity': u_vel, 'v_velocity': v_vel})

        # Convert midpoint data to a DataFrame
        midpoint_df = pd.DataFrame(midpoint_data)

        # Store the midpoint DataFrame for this point
        midpoint_dfs[point] = midpoint_df

    return midpoint_dfs

def resample_velocity_time_series(midpoint_dfs, months_per_bin=1):
    """
    Resample the midpoint velocity data for each point by aggregating over a specified number 
    of months per bin.
    
    Parameters:
      midpoint_dfs : dict
          Dictionary of DataFrames containing midpoint velocity data for each point. Each DataFrame
          must have a 'date' column that can be converted to a datetime.
      months_per_bin : int
          Number of months per aggregated bin. For example:
            - 1 for monthly (default),
            - 4 for 4-month intervals,
            - 6 for 6-month intervals,
            - 12 for yearly.
    
    Returns:
      resampled_dfs : dict
          Dictionary of resampled DataFrames for each point.
          Each DataFrame will have the 'date' column (representing the bin’s month-end)
          and the aggregated velocity values (using median).
    """
    resampled_dfs = {}
    
    # Build a frequency string using month-end frequency with the specified multiple.
    # For example, if months_per_bin=1, freq_str='1ME' (same as 'ME'); if 4, then '4ME', etc.
    freq_str = f'{months_per_bin}ME'
    
    for point, df in midpoint_dfs.items():
        # Work on a copy of the DataFrame.
        df = df.copy()
        # Ensure that 'date' is a datetime column.
        df['date'] = pd.to_datetime(df['date'])
        # Set 'date' as the index for resampling.
        df.set_index('date', inplace=True)
        # Resample using the constructed frequency string and aggregate using the median.
        resampled_df = df.resample(freq_str).median().reset_index()
        resampled_dfs[point] = resampled_df
    
    return resampled_dfs


def prepare_csv_with_components(updated_dfs, geotiff_path):
    """
    Prepare two CSVs in EGMS-like format using velocity time series for each component (WE and NS).
    Include a 'pid', 'latitude', 'longitude', 'x', and 'y' columns (EPSG:4326),
    along with median velocities, RMSE, and time series data formatted as 'YYYYMMDD'.

    Parameters:
    - updated_dfs: Dictionary of DataFrames containing time series data for each point, 
                   including WE (u_velocity) and NS (v_velocity) components.
    - geotiff_path: Path to the GeoTIFF file used for georeferencing image coordinates.

    Returns:
    - formatted_csv_we: A Pandas DataFrame for the WE component, ready for CSV export.
    - formatted_csv_ns: A Pandas DataFrame for the NS component, ready for CSV export.
    """
    import random, string
    import rasterio
    from pyproj import Transformer
    import pandas as pd

    # Open the GeoTIFF file to get its transform and CRS
    with rasterio.open(geotiff_path) as src:
        transform = src.transform
        src_crs = src.crs
        # Create a transformer to convert from the GeoTIFF CRS to EPSG:4326
        transformer = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)

    def image_to_georef(x, y):
        """
        Convert image coordinates (col, row) to georeferenced coordinates in EPSG:4326.
        """
        x_coord, y_coord = rasterio.transform.xy(transform, y, x, offset='center')
        lon, lat = transformer.transform(x_coord, y_coord)
        return round(lon, 6), round(lat, 6)  # Round to 6 decimals for precision

    def generate_pid():
        """Generate a random alphanumeric PID."""
        return 'OT' + ''.join(random.choices(string.ascii_letters + string.digits, k=8))

    # Initialize lists for storing data for WE and NS components
    csv_data_we = []
    csv_data_ns = []

    # Loop through each point and its corresponding DataFrame
    for point, df in updated_dfs.items():
        # Convert image point to georeferenced coordinates in EPSG:4326
        lon, lat = image_to_georef(*point)
        pid = generate_pid()

        # Format dates in the required format 'DYYYYMMDD'
        df['formatted_date'] = 'D' + df['date'].dt.strftime('%Y%m%d')
        
        # Flatten the time series into dictionaries for WE and NS components.
        # Here, u_velocity corresponds to WE and v_velocity to NS.
        time_series_we = df.set_index('formatted_date')['u_velocity'].to_dict()
        time_series_ns = df.set_index('formatted_date')['v_velocity'].to_dict()
        
        # Limit decimal precision to 1 for WE and NS values
        time_series_we = {k: (round(v, 1) if isinstance(v, (int, float)) else v) for k, v in time_series_we.items()}
        time_series_ns = {k: (round(v, 1) if isinstance(v, (int, float)) else v) for k, v in time_series_ns.items()}
        
        median_velocity_we = round(df['u_velocity'].median(), 1)
        median_velocity_ns = round(df['v_velocity'].median(), 1)

        # Compute RMSE for both WE and NS components.
        # Convert timestamps from the DataFrame index to seconds since epoch.
        timestamps = pd.to_datetime(df.index).astype(int) / 1e9
        rmse_we = compute_rmse_polynomial(df['u_velocity'].values, timestamps, degree=3)
        rmse_ns = compute_rmse_polynomial(df['v_velocity'].values, timestamps, degree=3)

        # Prepare the row for the WE component
        row_we = {
            'pid': pid,                      # Unique identifier for the point
            'latitude': lat,                 # Latitude in EPSG:4326
            'longitude': lon,                # Longitude in EPSG:4326
            'x': point[0],                   # Image X-coordinate
            'y': point[1],                   # Image Y-coordinate
            'median_velocity': median_velocity_we,  # Median velocity for WE
            'rmse': round(rmse_we, 2)        # RMSE for WE
        }
        row_we.update(time_series_we)  # Add WE time series data
        csv_data_we.append(row_we)

        # Prepare the row for the NS component
        row_ns = {
            'pid': pid,                      # Unique identifier for the point
            'latitude': lat,                 # Latitude in EPSG:4326
            'longitude': lon,                # Longitude in EPSG:4326
            'x': point[0],                   # Image X-coordinate
            'y': point[1],                   # Image Y-coordinate
            'median_velocity': median_velocity_ns,  # Median velocity for NS
            'rmse': round(rmse_ns, 2)        # RMSE for NS
        }
        row_ns.update(time_series_ns)  # Add NS time series data
        csv_data_ns.append(row_ns)

    # Convert the lists of dictionaries to DataFrames
    formatted_csv_we = pd.DataFrame(csv_data_we)
    formatted_csv_ns = pd.DataFrame(csv_data_ns)

    return formatted_csv_we, formatted_csv_ns

def iterative_weighted_monthly_velocity(displacement_data, velocity_component='u_velocity', 
                                          max_iter=10, tol=1e-3, smoothing_window=3,
                                          min_snr=None, min_pkr=None,
                                          months_per_bin=1):
    """
    Iteratively estimate velocities for each point from the distributed displacement data,
    aggregating the output over a specified number of months per bin.
    
    After convergence, a rolling median smoothing is applied to reduce noise relative to neighboring bins.
    Additionally, contributions for each measurement are normalized after each iteration so that the sum
    of contributions equals the original measurement value.
    
    Parameters:
      displacement_data : dict
          Dictionary where each key is a point (e.g., (x, y)) and each value is a dictionary
          with keys: 'dat1', 'dat2', 'u_velocity', 'v_velocity', 'pkrs', and 'snrs'. Each entry should
          be a list of equal length.
      velocity_component : str
          Which velocity component to use ('u_velocity' or 'v_velocity').
      max_iter : int
          Maximum number of iterations.
      tol : float
          Convergence tolerance (maximum change in bin estimates).
      smoothing_window : int
          Window size (in bins) for the rolling median filter applied after convergence.
      min_snr : float or None
          Minimum acceptable snr. Measurements with snr below this value will be skipped.
      min_pkr : float or None
          Minimum acceptable pkr. Measurements with pkr below this value will be skipped.
      months_per_bin : int
          Number of months per aggregated bin. (1 for monthly, 4 for 4-month intervals, 6 for 6-month, 12 for annual.)
      
    Returns:
      estimates_dict : dict
          Dictionary mapping each point to a DataFrame with columns:
             'month'             - Timestamp for the bin start,
             'velocity'          - Raw estimated velocity for that bin,
             'smoothed_velocity' - Velocity after applying rolling median smoothing.
    """
    
    estimates_dict = {}
    
    # Process each point separately
    for point, data in displacement_data.items():
        # Extract data lists
        dat1_list = data['dat1']
        dat2_list = data['dat2']
        velocities = data[velocity_component]
        pkrs = data.get('pkrs', [np.nan] * len(dat1_list))
        snrs = data.get('snrs', [np.nan] * len(dat1_list))
        
        # Convert dates to pandas Timestamps
        dat1_list = [pd.Timestamp(d) for d in dat1_list]
        dat2_list = [pd.Timestamp(d) for d in dat2_list]
        
        # Build a list of valid measurements (skip those with invalid velocity or failing quality criteria)
        measurements = []
        for d1, d2, v, pkr, snr in zip(dat1_list, dat2_list, velocities, pkrs, snrs):
            if np.isnan(v):
                continue
            if min_snr is not None and (np.isnan(snr) or snr < min_snr):
                continue
            if min_pkr is not None and (np.isnan(pkr) or pkr < min_pkr):
                continue
            T = (d2 - d1).total_seconds() / 86400.0  # total interval in days
            if T <= 0:
                continue
            measurements.append({'dat1': d1, 'dat2': d2, 'v': v, 'T': T, 'pkr': pkr, 'snr': snr})
        
        # If no valid measurements, return an empty DataFrame.
        if len(measurements) == 0:
            estimates_dict[point] = pd.DataFrame(columns=['month', 'velocity', 'smoothed_velocity'])
            continue
        
        # Determine overall time span for this point (using bin start dates)
        overall_start = min(m['dat1'] for m in measurements).to_period('M').to_timestamp()
        overall_end = max(m['dat2'] for m in measurements).to_period('M').to_timestamp() + pd.offsets.MonthEnd(0)
        # Use frequency like 'MS' for monthly, '4MS' for every 4 months, etc.
        freq_str = f'{months_per_bin}MS'
        bin_index = pd.date_range(start=overall_start, end=overall_end, freq=freq_str)
        
        # Compute bin end for each bin: bin_end = bin_start + months_per_bin months - 1 day.
        bin_intervals = {b: b + pd.DateOffset(months=months_per_bin) - pd.Timedelta(days=1) for b in bin_index}
        
        # For each measurement, compute its overlap (in days) with each bin and derive weights.
        # contributions[i][bin] = {'w': weight, 'c': initial contribution}
        contributions = {}
        for i, m in enumerate(measurements):
            contributions[i] = {}
            for b in bin_index:
                b_end = bin_intervals[b]
                overlap_start = max(m['dat1'], b)
                overlap_end = min(m['dat2'], b_end)
                overlap = (overlap_end - overlap_start).total_seconds() / 86400.0
                if overlap > 0:
                    w = overlap / m['T']
                    contributions[i][b] = {'w': w, 'c': m['v'] * w}
        
        # Compute initial bin estimates by weighted averaging
        bin_estimates = {}
        for b in bin_index:
            num, den = 0.0, 0.0
            for i in contributions:
                if b in contributions[i]:
                    num += contributions[i][b]['c']
                    den += contributions[i][b]['w']
            bin_estimates[b] = num / den if den > 0 else np.nan
        
        # Iterative refinement of bin estimates
        for iteration in range(max_iter):
            errors = {}
            for i, m in enumerate(measurements):
                pred = 0.0
                for b, vals in contributions[i].items():
                    pred += vals['w'] * bin_estimates[b]
                errors[i] = m['v'] - pred
            
            for i in contributions:
                for b, vals in contributions[i].items():
                    delta = vals['w'] * errors[i]
                    contributions[i][b]['c'] += delta
                total_c = sum(val['c'] for val in contributions[i].values())
                if total_c != 0:
                    factor = measurements[i]['v'] / total_c
                    for b in contributions[i]:
                        contributions[i][b]['c'] *= factor
            
            new_bin_estimates = {}
            for b in bin_index:
                num, den = 0.0, 0.0
                for i in contributions:
                    if b in contributions[i]:
                        num += contributions[i][b]['c']
                        den += contributions[i][b]['w']
                new_bin_estimates[b] = num / den if den > 0 else np.nan
            
            diffs = [abs(new_bin_estimates[b] - bin_estimates[b])
                     for b in bin_index
                     if not np.isnan(new_bin_estimates[b]) and not np.isnan(bin_estimates[b])]
            max_diff = max(diffs) if diffs else 0
            bin_estimates = new_bin_estimates
            if max_diff < tol:
                print(f"Point {point}: Converged after {iteration+1} iterations (max change {max_diff}).")
                break
        else:
            print(f"Point {point}: Reached maximum iterations ({max_iter}) with max change {max_diff}.")
        
        # Create a DataFrame for the current point's bin estimates.
        df_bins = pd.DataFrame({
            'month': list(bin_estimates.keys()),
            'velocity': list(bin_estimates.values())
        }).sort_values('month').reset_index(drop=True)
        
        # Apply temporal smoothing with a rolling median filter.
        df_bins['smoothed_velocity'] = df_bins['velocity'].rolling(window=smoothing_window, 
                                                                    center=True, min_periods=1).median()
        estimates_dict[point] = df_bins
    
    return estimates_dict

def merge_monthly_estimates_with_format(monthly_estimates_u, monthly_estimates_v, use_smoothed=False):
    """
    Merge monthly estimates for u and v velocities into a single DataFrame for each point.
    The output DataFrame for each point contains the columns:
      - 'date': the month-end date,
      - 'u_velocity': the u velocity,
      - 'v_velocity': the v velocity,
      - 'formatted_date': a string formatted as 'DYYYYMMDD'.
      
    Parameters:
      monthly_estimates_u : dict
          Dictionary where each key is a point (e.g., (x, y)) and each value is a DataFrame
          with columns ['month', 'velocity'] or optionally ['month', 'velocity', 'smoothed_velocity'] for u velocities.
      monthly_estimates_v : dict
          Dictionary where each key is a point (e.g., (x, y)) and each value is a DataFrame
          with columns ['month', 'velocity'] or optionally ['month', 'velocity', 'smoothed_velocity'] for v velocities.
      use_smoothed : bool
          If True, use the 'smoothed_velocity' column (if present) instead of the original 'velocity'.
    
    Returns:
      merged_estimates : dict
          Dictionary where each key is a point and each value is a DataFrame with the columns:
          'date', 'u_velocity', 'v_velocity', 'formatted_date'.
    """
    import pandas as pd
    
    merged_estimates = {}
    # Create a union of all points from both dictionaries.
    all_points = set(monthly_estimates_u.keys()) | set(monthly_estimates_v.keys())
    
    for point in all_points:
        # Get the u and v DataFrames; if missing, use an empty DataFrame.
        u_df = monthly_estimates_u.get(point, pd.DataFrame(columns=['month', 'velocity'])).copy()
        v_df = monthly_estimates_v.get(point, pd.DataFrame(columns=['month', 'velocity'])).copy()
        
        # Select the appropriate column for u velocities.
        if use_smoothed and 'smoothed_velocity' in u_df.columns:
            u_df = u_df.rename(columns={'smoothed_velocity': 'u_velocity'})
            if 'velocity' in u_df.columns:
                u_df = u_df.drop(columns=['velocity'])
        elif 'velocity' in u_df.columns:
            u_df = u_df.rename(columns={'velocity': 'u_velocity'})
        
        # Select the appropriate column for v velocities.
        if use_smoothed and 'smoothed_velocity' in v_df.columns:
            v_df = v_df.rename(columns={'smoothed_velocity': 'v_velocity'})
            if 'velocity' in v_df.columns:
                v_df = v_df.drop(columns=['velocity'])
        elif 'velocity' in v_df.columns:
            v_df = v_df.rename(columns={'velocity': 'v_velocity'})
        
        # Merge the two DataFrames on the 'month' column (using outer join).
        merged_df = pd.merge(u_df, v_df, on='month', how='outer')
        merged_df.sort_values('month', inplace=True)
        
        # Ensure the 'month' column is in datetime format.
        merged_df['month'] = pd.to_datetime(merged_df['month'])
        
        # Convert the 'month' column (assumed to be month-start) to month-end dates.
        merged_df['date'] = merged_df['month'] + pd.offsets.MonthEnd(0)
        
        # Create a new column 'formatted_date' as 'DYYYYMMDD'
        merged_df['formatted_date'] = 'D' + merged_df['date'].dt.strftime('%Y%m%d')
        
        # Reorder columns to match desired output.
        merged_df = merged_df[['date', 'u_velocity', 'v_velocity', 'formatted_date']]
        merged_estimates[point] = merged_df.reset_index(drop=True)
    
    return merged_estimates

def estimate_velocity_time_series(displacement_data, method='iterative', 
                                  months_per_bin=4, max_iter=20, tol=1e-4, 
                                  smoothing_window=3, min_snr=3, min_pkr=1.3, 
                                  use_smoothed=True):
    """
    Estimate velocity time series from displacement data using one of two methods.
    
    Parameters:
      displacement_data : dict
          Dictionary where each key is a point (e.g., (x, y)) and each value is a dictionary
          with keys 'dat1', 'dat2', 'u_velocity', 'v_velocity', 'pkrs', and 'snrs'.
      method : str
          Which method to use. 'simple' uses midpoints and resampling.
          'iterative' uses an iterative weighted approach with quality filtering.
      months_per_bin : int
          Number of months per aggregated bin (e.g., 1 for monthly, 4 for 4-month intervals).
      max_iter : int
          Maximum number of iterations for the iterative method.
      tol : float
          Convergence tolerance for the iterative method.
      smoothing_window : int
          Window size (in bins) for the rolling median smoothing.
      min_snr : float
          Minimum acceptable snr for filtering measurements (iterative method).
      min_pkr : float
          Minimum acceptable pkr for filtering measurements (iterative method).
      use_smoothed : bool
          If True, use the smoothed estimates in the iterative method.
    
    Returns:
      velocity_estimates : dict
          Dictionary mapping each point to a DataFrame containing the estimated velocities.
          The DataFrame will have a time column (e.g., 'month' or 'date') and velocity columns.
    """
    
    if method.lower() == 'midpoint':
        # Method 1: Simple approach using midpoint dates and then resampling.
        midpoint_dfs = create_velocity_time_series(displacement_data)
        velocity_estimates = resample_velocity_time_series(midpoint_dfs, months_per_bin=months_per_bin)
        print("Using midpoint method (midpoint/resampling).")
    
    elif method.lower() == 'iterative':
        # Method 2: Iterative weighted approach for u and v components separately.
        monthly_estimates_u = iterative_weighted_monthly_velocity(
            displacement_data, velocity_component='u_velocity',
            max_iter=max_iter, tol=tol, smoothing_window=smoothing_window,
            min_snr=min_snr, min_pkr=min_pkr, months_per_bin=months_per_bin)
        
        monthly_estimates_v = iterative_weighted_monthly_velocity(
            displacement_data, velocity_component='v_velocity',
            max_iter=max_iter, tol=tol, smoothing_window=smoothing_window,
            min_snr=min_snr, min_pkr=min_pkr, months_per_bin=months_per_bin)
        
        velocity_estimates = merge_monthly_estimates_with_format(
            monthly_estimates_u, monthly_estimates_v, use_smoothed=use_smoothed)
        print("Using iterative weighted method with quality filtering and smoothing.")
    
    else:
        raise ValueError("Invalid method. Choose 'simple' or 'iterative'.")
    
    return velocity_estimates

def plot_fastest_points_components(csv_data_ew, csv_data_sn, top_n=5):
    """
    Plot the time series of the fastest points based on median velocity for both EW and NS components,
    showing both components for the same point on the same plot.

    Parameters:
    - csv_data_ew: Pandas DataFrame containing EW-component time series data and median velocity.
    - csv_data_sn: Pandas DataFrame containing NS-component time series data and median velocity.
    - top_n: Number of fastest points to plot for each component.
    """
    # Sort the data by median_velocity in descending order to find the fastest points
    fastest_points_ew = csv_data_ew.sort_values(by='median_velocity', ascending=False).head(top_n)

    for _, row_ew in fastest_points_ew.iterrows():
        point = row_ew['pid']
        median_velocity_ew = row_ew['median_velocity']

        # Find the corresponding row in SN-component DataFrame
        row_sn = csv_data_sn[csv_data_sn['pid'] == point]
        if row_sn.empty:
            print(f"Skipping point {point}: No corresponding NS data found.")
            continue
        row_sn = row_sn.iloc[0]  # Extract first matching row

        median_velocity_sn = row_sn['median_velocity']

        # Extract time series data (column names follow "DYYYYMMDD" format)
        time_series_ew = row_ew.filter(regex=r'^D\d{8}$')
        time_series_sn = row_sn.filter(regex=r'^D\d{8}$')

        # Convert date columns to datetime (removing leading 'D')
        dates_ew = pd.to_datetime(time_series_ew.index.str[1:], format='%Y%m%d')
        values_ew = time_series_ew.values

        dates_sn = pd.to_datetime(time_series_sn.index.str[1:], format='%Y%m%d')
        values_sn = time_series_sn.values

        # Plot the time series for EW and SN components
        plt.figure(figsize=(12, 6))
        plt.scatter(dates_ew, values_ew, label=f'EW-Component (Median Velocity: {median_velocity_ew:.2f} m/year)', marker='o')
        plt.scatter(dates_sn, values_sn, label=f'NS-Component (Median Velocity: {median_velocity_sn:.2f} m/year)', marker='x')
        plt.title(f'Time Series for Point {point}')
        plt.xlabel('Date')
        plt.ylabel('Displacement (m/year)')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()
