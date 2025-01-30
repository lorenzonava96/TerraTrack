import numpy as np
import pandas as pd
import rasterio
import random
import string
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial

def accumulate_displacement_with_placeholders(all_u, all_v, all_feature_points, separation, median_feature_points, dat1, dat2, pixel_size):
    """
    Accumulate displacements for each filtered feature point (median_feature_points) across all pairs,
    inserting NaN for missing intervals. Displacements are converted to meters using pixel size.
    Velocities are also computed for each interval.

    Parameters:
    - all_u: List of arrays containing u displacements in pixels.
    - all_v: List of arrays containing v displacements in pixels.
    - all_feature_points: List of arrays containing feature point coordinates.
    - separation: List of temporal separations for each pair (in years).
    - median_feature_points: Array of filtered feature points (reference grid).
    - dat1: Array of start dates for each interval.
    - dat2: Array of end dates for each interval.
    - pixel_size: Size of one pixel in meters.

    Returns:
    - displacement_data: Dictionary with aligned u_values, v_values, years_diff, dat1, dat2,
      u_velocity, and v_velocity for filtered points.
    """

    # Ensure points are consistently tuples
    median_feature_points = [tuple(fp) for fp in median_feature_points]

    # Initialize the data structure for filtered points only
    displacement_data = {tuple(fp): {
        'u_values': np.full(len(dat1), np.nan),
        'v_values': np.full(len(dat1), np.nan),
        'years_diff': np.full(len(dat1), np.nan),
        'dat1': [None] * len(dat1),
        'dat2': [None] * len(dat1),
        'u_velocity': np.full(len(dat1), np.nan),
        'v_velocity': np.full(len(dat1), np.nan),
    } for fp in median_feature_points}

    # Debugging: Ensure all arrays have matching lengths
    if not (len(all_u) == len(all_v) == len(all_feature_points) == len(separation) == len(dat1) == len(dat2)):
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

                # Calculate velocities (m/year) if year_diff is valid
                if not np.isnan(year_diff) and year_diff > 0:
                    u_velocity = u_meters / year_diff
                    v_velocity = v_meters / year_diff
                    displacement_data[point]['u_velocity'][i] = u_velocity
                    displacement_data[point]['v_velocity'][i] = v_velocity
                    
    return displacement_data

def create_velocity_time_series(displacement_data, interval='monthly'):
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

def resample_velocity_time_series(midpoint_dfs, interval='monthly'):
    """
    Resample the midpoint velocity data for each point by aggregating it monthly or yearly.

    Parameters:
    - midpoint_dfs: Dictionary of DataFrames containing midpoint velocity data for each point.
    - interval: Aggregation interval ('monthly' or 'yearly').

    Returns:
    - resampled_dfs: Dictionary of resampled DataFrames for each point.
    """
    resampled_dfs = {}

    for point, df in midpoint_dfs.items():
        # Ensure 'date' is the index for resampling
        df.set_index('date', inplace=True)

        if interval == 'monthly':
            # Resample by month and aggregate using the mean
            resampled_df = df.resample('ME').median().reset_index()
        elif interval == 'yearly':
            # Resample by year and aggregate using the mean
            resampled_df = df.resample('YE').median().reset_index()
        else:
            raise ValueError("Interval must be 'monthly' or 'yearly'.")

        # Store the resampled DataFrame for this point
        resampled_dfs[point] = resampled_df

    return resampled_dfs

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

def prepare_csv_with_components(updated_dfs, geotiff_path):
    """
    Prepare two CSVs in EGMS-like format using velocity time series for each component (EW and SN).
    Include a 'pid', 'latitude', 'longitude', 'x', and 'y' columns (EPSG:4326),
    along with median velocities, RMSE, and time series data formatted as 'YYYYMMDD'.

    Parameters:
    - updated_dfs: Dictionary of DataFrames containing time series data for each point, including EW and SN components.
    - geotiff_path: Path to the GeoTIFF file used for georeferencing image coordinates.

    Returns:
    - formatted_csv_ew: A Pandas DataFrame for the EW component, ready for CSV export.
    - formatted_csv_sn: A Pandas DataFrame for the SN component, ready for CSV export.
    """
    # Open the GeoTIFF file to get geotransform
    with rasterio.open(geotiff_path) as src:
        transform = src.transform

    def image_to_georef(x, y):
        """Convert image coordinates (col, row) to georeferenced coordinates."""
        lon, lat = rasterio.transform.xy(transform, y, x, offset='center')
        return round(lon, 6), round(lat, 6)  # Round to 6 decimals for precision

    def generate_pid():
        """Generate a random alphanumeric PID."""
        return 'OT' + ''.join(random.choices(string.ascii_letters + string.digits, k=8))

    # Initialize the lists for storing all points' data for EW and SN components
    csv_data_ew = []
    csv_data_sn = []

    # Loop through each point and its corresponding DataFrame
    for point, df in updated_dfs.items():
        # Convert point to georeferenced coordinates
        lon, lat = image_to_georef(*point)
        pid = generate_pid()

        # Format dates in the required format D YYYYMMDD
        df['formatted_date'] = 'D' + df['date'].dt.strftime('%Y%m%d')
        
        # Flatten the time series into dictionaries for EW and SN components
        time_series_ew = df.set_index('formatted_date')['u_velocity'].to_dict()
        time_series_sn = df.set_index('formatted_date')['v_velocity'].to_dict()
        
        # Limit decimal precision to 1 after the comma for EW and SN values
        time_series_ew = {k: (round(v, 1) if isinstance(v, (int, float)) else v) for k, v in time_series_ew.items()}
        time_series_sn = {k: (round(v, 1) if isinstance(v, (int, float)) else v) for k, v in time_series_sn.items()}
        
        median_velocity_ew = round(df['u_velocity'].median(), 1)
        median_velocity_sn = round(df['v_velocity'].median(), 1)

        # Compute RMSE for both EW and SN components
        timestamps = pd.to_datetime(df.index).astype(int) / 1e9  # Convert to seconds since epoch
        rmse_ew = compute_rmse_polynomial(df['u_velocity'].values, timestamps, degree=3)
        rmse_sn = compute_rmse_polynomial(df['u_velocity'].values, timestamps, degree=3)

        # Prepare the row for the EW component
        row_ew = {
            'pid': pid,                      # Unique alphanumeric identifier for the point
            'latitude': lat,                 # Latitude in EPSG:4326
            'longitude': lon,                # Longitude in EPSG:4326
            'x': point[0],                   # Image X-coordinate
            'y': point[1],                   # Image Y-coordinate
            'median_velocity': median_velocity_ew,  # Add the median velocity for EW
            'rmse': round(rmse_ew, 2)        # Add RMSE for EW
        }
        row_ew.update(time_series_ew)  # Add time series data for EW
        csv_data_ew.append(row_ew)

        # Prepare the row for the SN component
        row_sn = {
            'pid': pid,                      # Unique alphanumeric identifier for the point
            'latitude': lat,                 # Latitude in EPSG:4326
            'longitude': lon,                # Longitude in EPSG:4326
            'x': point[0],                   # Image X-coordinate
            'y': point[1],                   # Image Y-coordinate
            'median_velocity': median_velocity_sn,  # Add the median velocity for SN
            'rmse': round(rmse_sn, 2)        # Add RMSE for SN
        }
        row_sn.update(time_series_sn)  # Add time series data for SN
        csv_data_sn.append(row_sn)

    # Convert the lists of dictionaries to DataFrames
    formatted_csv_ew = pd.DataFrame(csv_data_ew)
    formatted_csv_sn = pd.DataFrame(csv_data_sn)

    return formatted_csv_ew, formatted_csv_sn

def plot_fastest_points_components(csv_data_ew, csv_data_sn, top_n=5):
    """
    Plot the time series of the fastest points based on median velocity for both EW and SN components,
    showing both components for the same point on the same plot.

    Parameters:
    - csv_data_ew: Pandas DataFrame containing EW-component time series data and median velocity.
    - csv_data_sn: Pandas DataFrame containing SN-component time series data and median velocity.
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
            print(f"Skipping point {point}: No corresponding SN data found.")
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
        plt.scatter(dates_sn, values_sn, label=f'SN-Component (Median Velocity: {median_velocity_sn:.2f} m/year)', marker='x')
        plt.title(f'Time Series for Point {point}')
        plt.xlabel('Date')
        plt.ylabel('Displacement (m/year)')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

