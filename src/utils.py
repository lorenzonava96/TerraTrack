import os
import numpy as np

def handle_predictions(
    output_dir, method, match_func, results=None, separation=None, orig=None, dat1=None, dat2=None, save=True, load=True
):
    """
    Handles saving or loading prediction results based on the flags. If loading fails,
    data is created from the provided results and workspace variables and then saved.

    Parameters:
        output_dir (str): Directory where results are saved or loaded from.
        method (str): Name of the method used (e.g., block_matching).
        match_func (str): Name of the matching function used.
        results (list): List of dictionaries containing prediction results.
        separation (np.array): Separation vector.
        orig (np.array): Original study area image.
        dat1 (list): List of start dates.
        dat2 (list): List of end dates.
        save (bool): Flag to enable saving (ignored; saving happens in any case).
        load (bool): Flag to enable loading.

    Returns:
        dict: Prediction data loaded from the file or created from the results.
    """
    file_path = os.path.join(output_dir, f"{output_dir}_displacement_results_{method}_{match_func}.npz")

    if load:
        try:
            # Attempt to load the data
            loaded_data = np.load(file_path, allow_pickle=True)
            print(f"Loaded data successfully from {file_path}.")
            return {
                "all_u": list(loaded_data['all_u']),
                "all_v": list(loaded_data['all_v']),
                "all_feature_points": list(loaded_data['all_feature_points']),
                "all_pkrs": list(loaded_data['all_pkrs']),
                "all_snrs": list(loaded_data['all_snrs']),
                "separation": loaded_data['separation'],
                "study_area_image": loaded_data['study_area_image'],
                "dat1": loaded_data['dat1'],
                "dat2": loaded_data['dat2']
            }
        except FileNotFoundError:
            print(f"File not found: {file_path}. Switching to data creation from workspace.")

    # Generate data from workspace variables without checking for missing ones.
    print("Generating data from workspace variables...")

    # Initialize lists to store results
    all_u, all_v, all_feature_points = [], [], []
    all_pkrs, all_snrs = [], []

    # Process the results to populate data
    for result in results:
        all_u.append(result['u'])
        all_v.append(result['v'])
        all_feature_points.append(result['feature_points'])
        all_pkrs.append(result['pkrs'])
        all_snrs.append(result['snrs'])

    data = {
        "all_u": np.array(all_u, dtype=object),
        "all_v": np.array(all_v, dtype=object),
        "all_feature_points": np.array(all_feature_points, dtype=object),
        "all_pkrs": np.array(all_pkrs, dtype=object),
        "all_snrs": np.array(all_snrs, dtype=object),
        "separation": separation,
        "study_area_image": orig[..., 0],
        "dat1": dat1,
        "dat2": dat2
    }

    # Ensure the output directory exists and save results to a compressed NumPy file.
    os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(
        file_path,
        all_u=data["all_u"],
        all_v=data["all_v"],
        all_feature_points=data["all_feature_points"],
        all_pkrs=data["all_pkrs"],
        all_snrs=data["all_snrs"],
        separation=data["separation"],
        study_area_image=data["study_area_image"],
        dat1=data["dat1"],
        dat2=data["dat2"]
    )

    print(f"Results saved in '{output_dir}' directory:")
    print(f"- Compressed NumPy file: {file_path}")

    return data

def play_alert():
    """
    Plays a short trilling alert tone — useful as a simple notification sound.
    """
    import numpy as np
    from IPython.display import Audio

    fs = 44100       
    duration = 1      
    f1, f2 = 880, 660 

    t = np.linspace(0, duration, int(fs * duration), False)
    trill_pattern = ((np.floor(t * 10) % 2) == 0)  

    freq = np.where(trill_pattern, f1, f2)
    signal = np.sin(2 * np.pi * freq * t)

    return Audio(signal, rate=fs, autoplay=True)
