import numpy as np
import pandas as pd
import rasterio
import random
import string
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial
import random, string
import rasterio
from pyproj import Transformer
import pandas as pd
from skimage.transform import resize
import imageio.v2 as imageio
import io

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
    Prepare three CSVs in EGMS-like format using velocity time series for each component:
    WE (u_velocity), NS (v_velocity), and the magnitude at each time step.
    
    Each CSV will include columns: 'pid', 'latitude', 'longitude', 'x', 'y', median velocities,
    RMSE, and time series data with dates formatted as 'DYYYYMMDD'.
    
    Parameters:
      updated_dfs: Dictionary of DataFrames containing time series data for each point,
                   including u_velocity and v_velocity columns.
      geotiff_path: Path to the GeoTIFF file used for georeferencing image coordinates.
      
    Returns:
      formatted_csv_we: DataFrame for the WE component.
      formatted_csv_ns: DataFrame for the NS component.
      formatted_csv_mag: DataFrame for the magnitude component.
    """
    
    # Open the GeoTIFF file to get its transform and CRS.
    with rasterio.open(geotiff_path) as src:
        transform = src.transform
        src_crs = src.crs
        # Create a transformer to convert from the GeoTIFF CRS to EPSG:4326.
        transformer = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
    
    def image_to_georef(x, y):
        """
        Convert image coordinates (col, row) to georeferenced coordinates in EPSG:4326.
        """
        x_coord, y_coord = rasterio.transform.xy(transform, y, x, offset='center')
        lon, lat = transformer.transform(x_coord, y_coord)
        return round(lon, 6), round(lat, 6)
    
    def generate_pid():
        """Generate a random alphanumeric PID."""
        return 'OT' + ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    
    # Initialize lists to store data for WE, NS, and magnitude.
    csv_data_we = []
    csv_data_ns = []
    csv_data_mag = []
    
    # Loop through each point and its corresponding DataFrame.
    for point, df in updated_dfs.items():
        # Convert image coordinates to georeferenced coordinates.
        lon, lat = image_to_georef(*point)
        pid = generate_pid()
        
        # Format the dates into the desired 'DYYYYMMDD' format.
        df['formatted_date'] = 'D' + df['date'].dt.strftime('%Y%m%d')
        
        # Flatten the time series into dictionaries for each component.
        # WE (u_velocity) and NS (v_velocity)
        time_series_we = df.set_index('formatted_date')['u_velocity'].to_dict()
        time_series_ns = df.set_index('formatted_date')['v_velocity'].to_dict()
        
        # Compute the magnitude for each time step.
        df['magnitude'] = np.sqrt(df['u_velocity']**2 + df['v_velocity']**2)
        time_series_mag = df.set_index('formatted_date')['magnitude'].to_dict()
        
        # Limit decimal precision to 1 for WE, NS, and magnitude values.
        time_series_we = {k: (round(v, 1) if isinstance(v, (int, float)) else v) for k, v in time_series_we.items()}
        time_series_ns = {k: (round(v, 1) if isinstance(v, (int, float)) else v) for k, v in time_series_ns.items()}
        time_series_mag = {k: (round(v, 1) if isinstance(v, (int, float)) else v) for k, v in time_series_mag.items()}
        
        # Compute median velocities for WE, NS, and magnitude.
        median_velocity_we = round(df['u_velocity'].median(), 1)
        median_velocity_ns = round(df['v_velocity'].median(), 1)
        median_velocity_mag = round(df['magnitude'].median(), 1)
        
        # Compute RMSE for each component.
        # Timestamps are taken from the DataFrame's index (assuming it is a datetime index).
        timestamps = pd.to_datetime(df.index).astype(int) / 1e9
        rmse_we = compute_rmse_polynomial(df['u_velocity'].values, timestamps, degree=3)
        rmse_ns = compute_rmse_polynomial(df['v_velocity'].values, timestamps, degree=3)
        rmse_mag = compute_rmse_polynomial(df['magnitude'].values, timestamps, degree=3)
        
        # Prepare row for WE component.
        row_we = {
            'pid': pid,
            'latitude': lat,
            'longitude': lon,
            'x': point[0],
            'y': point[1],
            'median_velocity': median_velocity_we,
            'rmse': round(rmse_we, 2)
        }
        row_we.update(time_series_we)
        csv_data_we.append(row_we)
        
        # Prepare row for NS component.
        row_ns = {
            'pid': pid,
            'latitude': lat,
            'longitude': lon,
            'x': point[0],
            'y': point[1],
            'median_velocity': median_velocity_ns,
            'rmse': round(rmse_ns, 2)
        }
        row_ns.update(time_series_ns)
        csv_data_ns.append(row_ns)
        
        # Prepare row for Magnitude component.
        row_mag = {
            'pid': pid,
            'latitude': lat,
            'longitude': lon,
            'x': point[0],
            'y': point[1],
            'median_velocity': median_velocity_mag,
            'rmse': round(rmse_mag, 2)
        }
        row_mag.update(time_series_mag)
        csv_data_mag.append(row_mag)
    
    # Convert the lists of dictionaries into DataFrames.
    formatted_csv_we = pd.DataFrame(csv_data_we)
    formatted_csv_ns = pd.DataFrame(csv_data_ns)
    formatted_csv_mag = pd.DataFrame(csv_data_mag)
    
    return formatted_csv_we, formatted_csv_ns, formatted_csv_mag

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
        plt.figure(figsize=(6, 3))
        plt.scatter(dates_ew, values_ew, label=f'EW-Component (Median Velocity: {median_velocity_ew:.2f} m/year)', marker='o')
        plt.scatter(dates_sn, values_sn, label=f'NS-Component (Median Velocity: {median_velocity_sn:.2f} m/year)', marker='x')
        plt.title(f'Time Series for Point {point}')
        plt.xlabel('Date')
        plt.ylabel('Displacement (m/year)')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

def daily_velocity_series(displacement_data, velocity_component='u_velocity', 
                          min_snr=None, min_pkr=None):
    """
    Build a continuous daily velocity time series for each point.
    
    For each valid measurement (from an image pair), the velocity value is assumed to
    apply uniformly to every day between dat1 and dat2. In case multiple measurements
    contribute to the same day (overlap), the median is computed.
    
    Parameters:
      displacement_data : dict
          Dictionary where each key is a point (e.g., (x, y)) and each value is a dictionary with keys:
          'dat1', 'dat2', 'u_velocity', 'v_velocity', 'pkrs', and 'snrs'. All lists must have equal length.
      velocity_component : str
          Which velocity component to use ('u_velocity' or 'v_velocity').
      min_snr : float or None
          Minimum acceptable snr. Measurements with snr below this threshold are skipped.
      min_pkr : float or None
          Minimum acceptable pkr. Measurements with pkr below this threshold are skipped.
      
    Returns:
      daily_series_dict : dict
          Dictionary mapping each point to a DataFrame with columns:
             'date'     - Each day in the overall observation period,
             'velocity' - Median velocity for that day.
    """
    daily_series_dict = {}
    
    # Process each point separately.
    for point, data in displacement_data.items():
        # Convert the date lists to pandas Timestamps.
        dat1_list = [pd.Timestamp(d) for d in data['dat1']]
        dat2_list = [pd.Timestamp(d) for d in data['dat2']]
        velocities = data[velocity_component]
        pkrs = data.get('pkrs', [np.nan] * len(dat1_list))
        snrs = data.get('snrs', [np.nan] * len(dat1_list))
        
        # Build a list of valid measurements by applying quality filters.
        valid_measurements = []
        for d1, d2, v, pkr, snr in zip(dat1_list, dat2_list, velocities, pkrs, snrs):
            if np.isnan(v):
                continue
            if min_snr is not None and (np.isnan(snr) or snr < min_snr):
                continue
            if min_pkr is not None and (np.isnan(pkr) or pkr < min_pkr):
                continue
            # Only consider measurements with a positive duration.
            if d2 <= d1:
                continue
            valid_measurements.append((d1, d2, v))
        
        # If no valid measurements exist, return an empty DataFrame for this point.
        if not valid_measurements:
            daily_series_dict[point] = pd.DataFrame(columns=['date', 'velocity'])
            continue
        
        # Determine the overall time span for this point.
        overall_start = min(m[0] for m in valid_measurements)
        overall_end = max(m[1] for m in valid_measurements)
        # Create a complete daily date range covering the overall period.
        daily_index = pd.date_range(start=overall_start, end=overall_end, freq='D')
        
        # Create a dictionary to collect daily velocity contributions.
        # For each day, we will collect all velocity estimates that "cover" that day.
        daily_contrib = {day: [] for day in daily_index}
        
        # For each measurement, assign its velocity to every day in its interval.
        for d1, d2, v in valid_measurements:
            # Create a daily date range for the measurement (inclusive).
            days = pd.date_range(start=d1, end=d2, freq='D')
            for day in days:
                daily_contrib[day].append(v)
        
        # For each day in the overall period, compute the median velocity (if there are contributions).
        daily_values = []
        for day in daily_index:
            if daily_contrib[day]:
                daily_values.append(np.median(daily_contrib[day]))
            else:
                daily_values.append(np.nan)
        
        # Create a DataFrame for the daily time series.
        df_daily = pd.DataFrame({'date': daily_index, 'velocity': daily_values})
        daily_series_dict[point] = df_daily
    
    return daily_series_dict

def resample_daily_series(daily_series_dict, months_per_bin=1):
    """
    Resample the daily velocity series to a coarser time scale.
    
    Parameters:
      daily_series_dict : dict
          Dictionary mapping each point to a DataFrame with columns 'date' and 'velocity'.
      months_per_bin : int
          Number of months per aggregated bin (e.g., 1 for monthly, 2 for bi-monthly, 3 for quarterly, etc.).
          Note: For annual aggregation, use months_per_bin=12.
    
    Returns:
      resampled_dict : dict
          Dictionary mapping each point to a DataFrame with columns:
             'date'     - The bin’s end date (by default, month-end for monthly bins),
             'velocity' - The median velocity aggregated over the bin.
    """
    resampled_dict = {}
    
    for point, df in daily_series_dict.items():
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        # For annual aggregation, use frequency 'A'; otherwise, use f'{months_per_bin}M'
        if months_per_bin == 12:
            freq = 'YE'
        else:
            freq = f'{months_per_bin}ME'
        # Resample using the median as the aggregation function.
        res_df = df.resample(freq).median().reset_index()
        resampled_dict[point] = res_df
    
    return resampled_dict

def estimate_velocity_via_daily(displacement_data, velocity_component='u_velocity',
                                min_snr=None, min_pkr=None, months_per_bin=1):
    """
    Estimate a velocity time series for each point by first constructing a continuous daily time series
    (using the velocity estimate for the entire period between image pairs) and then resampling it into bins.
    
    Parameters:
      displacement_data : dict
          Input displacement data with keys: 'dat1', 'dat2', velocity component, 'pkrs', and 'snrs'.
      velocity_component : str
          Which velocity component to use ('u_velocity' or 'v_velocity').
      min_snr, min_pkr : float or None
          Quality thresholds for filtering measurements.
      months_per_bin : int
          Number of months per aggregated bin.
    
    Returns:
      resampled_dict : dict
          Dictionary mapping each point to a DataFrame with columns:
             'date'     - Bin end dates,
             'velocity' - Aggregated (median) velocity for that bin.
    """
    # First, build the daily velocity series.
    daily_series = daily_velocity_series(displacement_data, velocity_component=velocity_component,
                                         min_snr=min_snr, min_pkr=min_pkr)
    # Then, resample the daily series to the desired bin size.
    resampled_dict = resample_daily_series(daily_series, months_per_bin=months_per_bin)
    return resampled_dict

def create_velocity_time_series(displacement_data, min_snr=None, min_pkr=None):
    """
    Create velocity time series for each point by assigning velocities to the midpoint date
    between dat1 and dat2. Optionally, only measurements with a signal-to-noise ratio (snr) 
    above min_snr and a peak ratio (pkr) above min_pkr are used.
    
    Parameters:
      displacement_data : dict
          Dictionary where each key is a point (e.g., (x, y)) and each value is a dictionary
          with keys: 'dat1', 'dat2', 'u_velocity', 'v_velocity', and optionally 'snrs' and 'pkrs'.
          Each entry should be a list of equal length.
      min_snr : float or None
          Minimum acceptable snr. Measurements with snr below this value will be skipped.
      min_pkr : float or None
          Minimum acceptable pkr. Measurements with pkr below this value will be skipped.
          
    Returns:
      midpoint_dfs : dict
          Dictionary mapping each point to a DataFrame with columns:
             'date'       - The midpoint date,
             'u_velocity' - u velocity at the midpoint,
             'v_velocity' - v velocity at the midpoint.
    """
    
    midpoint_dfs = {}

    for point, data in displacement_data.items():
        midpoint_data = []
        
        # Extract data lists.
        dat1 = data['dat1']
        dat2 = data['dat2']
        u_velocities = data['u_velocity']
        v_velocities = data['v_velocity']
        # Optional quality metrics.
        snrs = data.get('snrs', [np.nan] * len(dat1))
        pkrs = data.get('pkrs', [np.nan] * len(dat1))
        
        # Process each measurement.
        for start, end, u_vel, v_vel, snr, pkr in zip(dat1, dat2, u_velocities, v_velocities, snrs, pkrs):
            # Check that velocities are valid.
            if np.isnan(u_vel) or np.isnan(v_vel):
                continue
            
            # Apply quality filters if thresholds are provided.
            if min_snr is not None:
                # If snr is missing or below threshold, skip.
                if np.isnan(snr) or snr < min_snr:
                    continue
            if min_pkr is not None:
                # If pkr is missing or below threshold, skip.
                if np.isnan(pkr) or pkr < min_pkr:
                    continue
            
            # Calculate midpoint date.
            start_date = pd.Timestamp(start)
            end_date = pd.Timestamp(end)
            midpoint_date = start_date + (end_date - start_date) / 2

            # Append the midpoint velocity data.
            midpoint_data.append({'date': midpoint_date, 
                                  'u_velocity': u_vel, 
                                  'v_velocity': v_vel})
        
        # Convert the collected data to a DataFrame.
        midpoint_df = pd.DataFrame(midpoint_data)
        
        # Store the DataFrame for this point.
        midpoint_dfs[point] = midpoint_df

    return midpoint_dfs

def estimate_velocity_time_series(displacement_data, method='weighted', 
                                  months_per_bin=4, min_snr=3, min_pkr=1.3):
    """
    Estimate velocity time series from displacement data using one of two methods.
    
    Two methods are available:
    
    - 'midpoint': A simple approach that assigns each image pair's velocity to the midpoint
      between dat1 and dat2, and then resamples these midpoint estimates into bins (e.g., monthly,
      quarterly, etc.).
    
    - 'weighted': A weighted approach that distributes each image pair's velocity uniformly over every
      day between dat1 and dat2, thereby constructing a continuous daily time series. Daily values
      are aggregated (using the median) into bins of the user-specified length. Quality filtering based on
      a minimum snr and a minimum pkr is applied prior to aggregation.
    
    Parameters:
      displacement_data : dict
          Dictionary where each key is a point (e.g., (x, y)) and each value is a dictionary
          with keys 'dat1', 'dat2', 'u_velocity', 'v_velocity', 'pkrs', and 'snrs'. Each list must have equal length.
      method : str
          The method to use for velocity estimation. Choose 'midpoint' for the simple midpoint/resampling approach,
          or 'weighted' for the approach that constructs a continuous daily time series with quality filtering.
      months_per_bin : int
          Number of months per aggregated bin (e.g., 1 for monthly, 4 for 4-month intervals, 6 for 6-month, or 12 for annual).
      min_snr : float
          Minimum acceptable signal-to-noise ratio for filtering measurements (used in the weighted method).
      min_pkr : float
          Minimum acceptable peak ratio for filtering measurements (used in the weighted method).
    
    Returns:
      velocity_estimates : dict
          Dictionary mapping each point to a DataFrame containing the estimated velocities. Each DataFrame
          will have a time column (e.g., 'month' or 'date') and velocity columns.
    """
    
    if method.lower() == 'midpoint':
        # Method 1: Simple approach using midpoint dates and then resampling.
        midpoint_dfs = create_velocity_time_series(displacement_data, min_snr=None, min_pkr=None)
        velocity_estimates = resample_velocity_time_series(midpoint_dfs, months_per_bin=months_per_bin)
        print("Using midpoint method (midpoint/resampling).")
    
    elif method.lower() == 'weighted':
        # Method 2: Weighted approach that builds a continuous daily time series from image pairs,
        # applies quality filtering (min_snr, min_pkr), and then aggregates the daily series.
        monthly_estimates_u = estimate_velocity_via_daily(displacement_data, velocity_component='u_velocity',
                                                          min_snr=min_snr, min_pkr=min_pkr, months_per_bin=months_per_bin)
        monthly_estimates_v = estimate_velocity_via_daily(displacement_data, velocity_component='v_velocity',
                                                          min_snr=min_snr, min_pkr=min_pkr, months_per_bin=months_per_bin)
        velocity_estimates = merge_monthly_estimates_with_format(monthly_estimates_u, monthly_estimates_v)
        print("Using weighted method with quality filtering and daily aggregation.")
    
    else:
        raise ValueError("Invalid method. Choose 'midpoint' or 'weighted'.")
    
    return velocity_estimates

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
        
    merged_estimates = {}
    # Create a union of all points from both dictionaries.
    all_points = set(monthly_estimates_u.keys()) | set(monthly_estimates_v.keys())
    
    for point in all_points:
        # Get the u and v DataFrames; if missing, use an empty DataFrame.
        u_df = monthly_estimates_u.get(point, pd.DataFrame(columns=['date', 'velocity'])).copy()
        v_df = monthly_estimates_v.get(point, pd.DataFrame(columns=['date', 'velocity'])).copy()
        
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
        
        # Merge the two DataFrames on the 'date' column (using outer join).
        merged_df = pd.merge(u_df, v_df, on='date', how='outer')
        merged_df.sort_values('date', inplace=True)
        
        # Ensure the 'date' column is in datetime format.
        merged_df['date'] = pd.to_datetime(merged_df['date'])
        
        # Convert the 'date' column (assumed to be month-start) to month-end dates.
        merged_df['date'] = merged_df['date'] + pd.offsets.MonthEnd(0)
        
        # Create a new column 'formatted_date' as 'DYYYYMMDD'
        merged_df['formatted_date'] = 'D' + merged_df['date'].dt.strftime('%Y%m%d')
        
        # Reorder columns to match desired output.
        merged_df = merged_df[['date', 'u_velocity', 'v_velocity', 'formatted_date']]
        merged_estimates[point] = merged_df.reset_index(drop=True)
    
    return merged_estimates

def get_cartesian_points_from_mask(mask, target_shape):
    """
    Resample a boolean mask to the target shape and return the cartesian (x, y)
    coordinates of all True pixels.

    Parameters:
      mask (np.ndarray): Original boolean mask.
      target_shape (tuple): Desired shape (height, width) to resample to.

    Returns:
      np.ndarray: An array of shape (N, 2) where each row is [x, y]. Here,
                  x corresponds to the column index and y to the row index.
    """
    # Resample the mask to target shape using nearest-neighbor interpolation.
    # 'order=0' ensures that boolean values are preserved.
    resampled_mask = resize(mask.astype(np.uint8), target_shape, order=0, 
                            preserve_range=True).astype(bool)
    
    # Get the row, col indices where the mask is True.
    rows, cols = np.where(resampled_mask)
    # Convert to (x, y) coordinates: x = column, y = row.
    points = np.column_stack((cols, rows))
    return points

def create_gif_with_background_and_colorbar(tif_path, study_area_image, output_gif, duration=1.0, cmap="viridis", alpha=0.6):
    """
    Create a GIF from a multi-band GeoTIFF where each band is overlaid on the 
    study_area_image as background. Only magnitude values ≥ 0.2 are shown; values
    below 0.2 are fully transparent. A colorbar (ranging from 0 to the overall maximum 
    magnitude) is added on the side.
    
    Parameters:
      tif_path (str): Path to the multi-band GeoTIFF (e.g., f"{output_dir}_magnitude_multiband.tif").
      study_area_image (np.ndarray): Background image array. Its resolution should match the composite.
      output_gif (str): Path where the output GIF will be saved.
      duration (float): Duration (in seconds) for each frame in the GIF.
      cmap (str): Colormap to use for the magnitude overlay.
      alpha (float): Transparency for the overlay (applied to visible values).
    """
    # First, compute the overall maximum value across all bands (ignoring nodata and zeros).
    with rasterio.open(tif_path) as src:
        num_bands = src.count
        nodata = src.nodata
        overall_max = 0
        for band in range(2, num_bands + 1):
            band_data = src.read(band)
            valid_data = band_data[(band_data != nodata) & (band_data != 0)]
            if valid_data.size > 0:
                current_max = valid_data.max()
                overall_max = max(overall_max, current_max)
    
    print(f"Overall maximum magnitude (ignoring zeros and nodata): {overall_max}")
    
    frames = []
    with rasterio.open(tif_path) as src:
        num_bands = src.count
        print(f"Found {num_bands} bands in {tif_path}.")
        for band in range(2, num_bands + 1):
            # Read the band data.
            band_data = src.read(band)
            # Mask out values below 0.2.
            # This will mask all values < 0.2 so they are not shown.
            masked_data = np.ma.masked_less(band_data, 0.2)
            
            # Get a copy of the colormap and set its "bad" (masked) color to be fully transparent.
            cmap_instance = plt.get_cmap(cmap).copy()
            cmap_instance.set_bad(color=(0, 0, 0, 0))
            
            # Create a figure.
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Display the study area image as the background.
            ax.imshow(study_area_image, cmap="gray")
            
            # Overlay the magnitude data.
            # We use vmin=0.2 so that the color scale starts at 0.2 (values below are masked).
            im = ax.imshow(masked_data, cmap=cmap_instance, vmin=0.2, vmax=overall_max, alpha=alpha)
            
            ax.axis("off")
            ax.set_title(f"Band {band}", fontsize=14)
            
            # Add a colorbar on the side.
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Magnitude", rotation=270, labelpad=15)
            
            # Save the figure to an in-memory buffer.
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            buf.seek(0)
            
            # Read the image from the buffer and add it to the list of frames.
            img = imageio.imread(buf)
            frames.append(img)
    
    # Save all frames as a GIF.
    imageio.mimsave(output_gif, frames, duration=duration)
    print(f"GIF saved to {output_gif}")
