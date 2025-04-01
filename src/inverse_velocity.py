import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore
from datetime import datetime
from sklearn.linear_model import LinearRegression # type: ignore
import matplotlib.pyplot as plt # type: ignore
from matplotlib.ticker import MaxNLocator # type: ignore

def estimate_failure_time(dates, inv_velocity, n_points_for_fit=5):
    """
    Estimate a 'time of failure' by fitting a linear model to the last n_points_for_fit
    points of inverse velocity vs. time. Returns the estimated failure date if it makes sense,
    otherwise None.
    """
    # Subset the last n_points_for_fit points
    subset_dates = dates[-n_points_for_fit:]
    subset_invv = inv_velocity[-n_points_for_fit:]
    
    # Filter out NaNs.
    subset_data = [(d, v) for d, v in zip(subset_dates, subset_invv) if not pd.isna(v)]
    if len(subset_data) < 2:
        return None
    
    subset_dates, subset_invv = zip(*subset_data)
    t0 = min(subset_dates)
    days_since_t0 = [(d - t0).days for d in subset_dates]
    
    model = LinearRegression()
    X = np.array(days_since_t0).reshape(-1, 1)
    y = np.array(subset_invv)
    model.fit(X, y)
    slope = model.coef_[0]
    intercept = model.intercept_
    
    # Only proceed if the slope is negative (indicating accelerating displacement)
    if slope < 0:
        t_failure = -intercept / slope
        if t_failure > 0:
            # Define a cutoff: if the estimated failure time is later than a reasonable date, skip it.
            cutoff_date = pd.Timestamp('2050-01-01')
            max_days = (cutoff_date - t0).days
            if t_failure > max_days:
                return None
            try:
                estimated_date = t0 + pd.Timedelta(days=t_failure)
            except Exception as e:
                return None
            return estimated_date
    return None

def plot_inverse_velocity_monthly(csv_path, output_folder=None, save_plots=False,
                                  n_points_for_fit_list=[5]):
    """
    Read a CSV with columns including 'pid' and date columns in the format DYYYYMMDD.
    The velocity data are first melted to long format, inverse velocity is computed,
    and then aggregated to monthly resolution using the median.
    A plot is created for each PID, with inverse velocity vs. time.
    
    For each PID, the function estimates a "time of failure" using each value
    in n_points_for_fit_list. The median of the valid estimated failure dates is then used to plot a single
    vertical line and is stored in the output.
    
    Parameters:
      csv_path : str
          Path to the CSV file.
      output_folder : str or None
          If provided and save_plots is True, plots will be saved as PNG files in this folder;
          otherwise, plots are displayed interactively.
      save_plots : bool
          If True, plots are saved (if output_folder is provided). Otherwise, they are displayed.
      n_points_for_fit_list : list of int
          List of sample sizes to use when fitting the linear regression for failure time estimation.
    
    Returns:
      failure_dates : list of dict
          A list of dictionaries, one per PID, with keys:
             'pid': the PID,
             'failure_date': the median estimated failure date.
    """
    df = pd.read_csv(csv_path)
    
    # Identify date columns (those starting with 'D')
    date_cols = [col for col in df.columns if col.startswith('D')]
    
    # Melt the DataFrame into long format: one row per (pid, date, velocity)
    melted = df.melt(id_vars=['pid'], value_vars=date_cols, var_name='date_str', value_name='velocity')
    
    # Remove the leading 'D' and convert date strings to datetime objects.
    melted['date'] = pd.to_datetime(melted['date_str'].str.replace('D', ''), format='%Y%m%d')
    
    # Compute inverse velocity (guard against division by zero)
    melted['inv_velocity'] = melted['velocity'].apply(lambda v: 1.0/v if v != 0 else np.nan)
    
    # Aggregate to monthly resolution using the median inverse velocity.
    melted['month'] = melted['date'].dt.to_period('M').dt.to_timestamp()
    monthly = melted.groupby(['pid', 'month'])['inv_velocity'].median().reset_index()
    
    failure_dates = []
    
    for pid in monthly['pid'].unique():
        df_pid = monthly[monthly['pid'] == pid].copy()
        df_pid.sort_values('month', inplace=True)
        
        dates = list(df_pid['month'])
        invv = list(df_pid['inv_velocity'])
        
        plt.figure(figsize=(8,5))
        plt.plot(dates, invv, marker='o', linestyle='-', color='b', label='Inverse Velocity')
        plt.title(f'Inverse Velocity vs. Time (Monthly) for PID: {pid}')
        plt.xlabel('Month')
        plt.ylabel('1 / Velocity')
        plt.grid(True)
        
        failure_estimates = []
        # Try different n_points_for_fit values.
        for n in n_points_for_fit_list:
            if len(dates) >= n:
                fd = estimate_failure_time(dates, invv, n_points_for_fit=n)
                if fd is not None:
                    failure_estimates.append(fd)
        
        if failure_estimates:
            # Compute the median failure date from the valid estimates.
            median_ns = np.median([fd.value for fd in failure_estimates])
            median_failure = pd.Timestamp(int(median_ns))
            plt.axvline(median_failure, color='r', linestyle='--',
                        label=f'Median Failure: {median_failure.date()}')
            plt.legend()
            failure_dates.append({'pid': pid, 'failure_date': median_failure})
        
        if save_plots and output_folder is not None:
            out_path = f"{output_folder}/{pid}_inverse_velocity_monthly.png"
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved plot for {pid} to {out_path}")
        else:
            plt.show()
    
    return failure_dates

def failure_date_statistics(failure_dates_list):
    """
    Compute statistics on a list of failure dates to determine the most probable failure month.
    
    This function expects each element in failure_dates_list to be a dictionary with keys:
      'pid' and 'failure_date' (a pandas Timestamp).
    
    Parameters:
      failure_dates_list : list of dict
          Each dictionary has keys 'pid' and 'failure_date'.
          
    Returns:
      month_counts : pandas Series
          A Series with monthly periods as index (in YYYY-MM format) and counts as values.
    """
    import pandas as pd
    
    # Collect all non-None failure dates.
    all_failure_dates = [item['failure_date'] for item in failure_dates_list 
                         if item.get('failure_date') is not None]
    
    if not all_failure_dates:
        print("No failure dates available.")
        return None
    
    # Convert the list of Timestamps into a pandas Series.
    dates_series = pd.Series(all_failure_dates)
    
    # Convert each timestamp to a monthly period (YYYY-MM)
    month_periods = dates_series.dt.to_period('M')
    
    # Count the frequency of each month and sort the result.
    month_counts = month_periods.value_counts().sort_index()
    
    # Determine the mode (the month with the highest count)
    most_common_month = month_counts.idxmax()
    print("Most common failure month:", most_common_month)
    
    return month_counts

def plot_failure_distribution(month_counts):
    """
    Plot the distribution of estimated failure months as a bar chart in a publication-quality style.
    
    Parameters:
      month_counts : pandas Series
          A Series where the index is the failure month (e.g., '2022-06') and the values
          are the counts (number of points with that estimated failure month).
    
    Returns:
      None. Displays the plot.
    """
    # Try to use the seaborn-whitegrid style; if it's not available, use default style.
    try:
        plt.style.use('seaborn-whitegrid')
    except Exception as e:
        print("Warning: 'seaborn-whitegrid' style not found; using default style.")
    
    # Convert the index to string labels for the x-axis.
    x_labels = month_counts.index.astype(str)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create the bar chart with black edges.
    bars = ax.bar(x_labels, month_counts.values, color='steelblue', edgecolor='black', linewidth=0.7)
    
    # Set axis labels.
    ax.set_xlabel('Estimated Failure Month', fontsize=14)
    ax.set_ylabel('Estimates Count', fontsize=14)
    
    # Limit the number of x-axis ticks.
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
    
    # Rotate x-axis tick labels for readability.
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    
    # # Add a vertical line at the target month (example: "2023-09").
    # target_month = "2023-09"
    # x_list = list(x_labels)
    # if target_month in x_list:
    #     pos = x_list.index(target_month)
    #     ax.axvline(x=pos, color='red', linestyle='--', label=f'Month of Failure: {target_month}')
    #     ax.legend(fontsize=12)
    
    plt.tight_layout()
    plt.show()
