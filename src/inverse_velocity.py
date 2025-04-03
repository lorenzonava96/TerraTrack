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
    
    # Filter out NaNs
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
            cutoff_date = pd.Timestamp('2100-01-01')
            max_days = (cutoff_date - t0).days
            if t_failure > max_days:
                return None
            try:
                estimated_date = t0 + pd.Timedelta(days=t_failure)
            except Exception as e:
                return None
            return estimated_date
    return None

def compute_inverse_velocity_failure_dates(csv_path, n_points_for_fit_list=[5]):
    """
    Read a CSV with columns including 'pid' and date columns in the format DYYYYMMDD.
    The velocity data are first melted to long format, inverse velocity is computed,
    and then aggregated to monthly resolution using the median.
    
    For each PID, the function attempts to estimate a "time of failure" using each value
    in n_points_for_fit_list (e.g., 4, 6). Each estimated failure date is stored in a dictionary.
    
    Parameters:
      csv_path : str
          Path to the CSV file.
      n_points_for_fit_list : list of int
          List of sample sizes to use when fitting the linear regression for failure time estimation.
    
    Returns:
      failure_dates : list of dict
          A list of dictionaries, one per PID, with keys:
             'pid': the PID,
             'failure_dates': a dictionary mapping each n_points value to the estimated failure date.
    """
    import pandas as pd
    import numpy as np
    
    df = pd.read_csv(csv_path)
    
    # Identify date columns (those starting with 'D')
    date_cols = [col for col in df.columns if col.startswith('D')]
    
    # Melt the DataFrame into long format: one row per (pid, date, velocity)
    melted = df.melt(id_vars=['pid'], value_vars=date_cols, 
                     var_name='date_str', value_name='velocity')
    
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
        
        failure_dict = {}
        # Try different n_points_for_fit values.
        for n in n_points_for_fit_list:
            if len(dates) >= n:
                fd = estimate_failure_time(dates, invv, n_points_for_fit=n)
                if fd is not None:
                    failure_dict[n] = fd
        
        failure_dates.append({'pid': pid, 'failure_dates': failure_dict})
    
    return failure_dates


def failure_date_statistics(failure_dates_list):
    """
    Compute statistics on a list of failure dates to determine the most probable failure month.
    
    This function expects each element in failure_dates_list to be a dictionary with keys:
      'pid' and 'failure_dates', where 'failure_dates' is a dictionary mapping sample sizes
      (e.g., 4, 6) to a failure date (a pandas Timestamp).
    
    Parameters:
      failure_dates_list : list of dict
          Each dictionary has keys 'pid' and 'failure_dates'. 'failure_dates' is itself a dict,
          e.g., {4: Timestamp(...), 6: Timestamp(...)}.
          
    Returns:
      month_counts : pandas Series
          A Series with monthly periods as index (in YYYY-MM format) and counts as values.
    """
    
    # Collect all failure dates from the nested dictionaries.
    all_failure_dates = []
    for item in failure_dates_list:
        fd_dict = item.get('failure_dates', {})
        # Loop over each sample size and its failure date
        for sample_size, fd in fd_dict.items():
            if fd is not None:
                all_failure_dates.append(fd)
    
    if not all_failure_dates:
        print("No failure dates available.")
        return None
    
    # Convert the list of Timestamps into a pandas Series.
    dates_series = pd.Series(all_failure_dates)
    
    # Convert each timestamp to a monthly period (YYYY-MM)
    month_periods = dates_series.dt.to_period('M')
    
    # Count the frequency of each month and sort by index.
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
    # Use a clean style for publication-quality figures.
    plt.style.use('seaborn-whitegrid')
    
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
    
    # Add vertical line at June 2022.
    # Since x_labels are categorical, find the index of "2022-06".
    x_list = list(x_labels)
    target_month = "2022-06"
    if target_month in x_list:
        pos = x_list.index(target_month)
        # ax.axvline(x=pos, color='red', linestyle='--', label=f'Month of Failure: {target_month}')
        # ax.legend(fontsize=12)
    
    # # Optionally, annotate each bar with its height (if desired)
    # for bar in bars:
    #     height = bar.get_height()
    #     ax.annotate(f'{height:.0f}',
    #                 xy=(bar.get_x() + bar.get_width() / 2, height),
    #                 xytext=(0, 5),  # vertical offset
    #                 textcoords="offset points",
    #                 ha='center', va='bottom', fontsize=12)
    
    plt.tight_layout()
    plt.show()
