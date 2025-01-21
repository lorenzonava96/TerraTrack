import numpy as np
import os

def handle_predictions(
    output_dir, method, match_func, results=None, separation=None, orig=None, dat1=None, dat2=None, save=True, load=True
):
    """
    Handles saving or loading prediction results based on the flags. If both saving and loading are disabled, data is created
    from the provided results and other variables in the workspace.

    Parameters:
        output_dir (str): Directory where results are saved or loaded from.
        method (str): Name of the method used (e.g., block_matching).
        match_func (str): Name of the matching function used.
        results (list): List of dictionaries containing prediction results (used for saving or creating data).
        separation (np.array): Separation vector (used for saving or creating data).
        orig (np.array): Original study area image (used for saving or creating data).
        dat1 (list): List of start dates (used for saving or creating data).
        dat2 (list): List of end dates (used for saving or creating data).
        save (bool): Flag to enable or disable saving.
        load (bool): Flag to enable or disable loading.

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
                "all_max_corrs": list(loaded_data['all_max_corrs']),
                "all_snrs": list(loaded_data['all_snrs']),
                "separation": loaded_data['separation'],
                "study_area_image": loaded_data['study_area_image'],
                "dat1": loaded_data['dat1'],
                "dat2": loaded_data['dat2']
            }
        except FileNotFoundError:
            print(f"File not found: {file_path}. Switching to data creation from workspace.")
            load = False  # Fallback to data creation if the file isn't found.

    if not load:
        # Validate required variables for data creation or saving
        missing_vars = [
            var_name
            for var_name, var_value in {
                "results": results,
                "separation": separation,
                "orig": orig,
                "dat1": dat1,
                "dat2": dat2
            }.items()
            if var_value is None
        ]
        if missing_vars:
            raise ValueError(f"Missing required variables: {', '.join(missing_vars)}")

        print("Generating data from workspace variables...")

        # Initialize lists to store results
        all_u, all_v, all_feature_points = [], [], []
        all_max_corrs, all_snrs = [], []

        # Process the results to populate data
        for result in results:
            all_u.append(result['u'])
            all_v.append(result['v'])
            all_feature_points.append(result['feature_points'])
            all_max_corrs.append(result['max_corrs'])
            all_snrs.append(result['snrs'])

        data = {
            "all_u": np.array(all_u, dtype=object),
            "all_v": np.array(all_v, dtype=object),
            "all_feature_points": np.array(all_feature_points, dtype=object),
            "all_max_corrs": np.array(all_max_corrs, dtype=object),
            "all_snrs": np.array(all_snrs, dtype=object),
            "separation": separation,
            "study_area_image": orig[..., 0],
            "dat1": dat1,
            "dat2": dat2
        }

        if save:
            # Directory to save results
            os.makedirs(output_dir, exist_ok=True)

            # Save results to a compressed NumPy file
            np.savez_compressed(
                file_path,
                all_u=data["all_u"],
                all_v=data["all_v"],
                all_feature_points=data["all_feature_points"],
                all_max_corrs=data["all_max_corrs"],
                all_snrs=data["all_snrs"],
                separation=data["separation"],
                study_area_image=data["study_area_image"],
                dat1=data["dat1"],
                dat2=data["dat2"]
            )

            print(f"Results saved in '{output_dir}' directory:")
            print(f"- Compressed NumPy file: {file_path}")

        return data

def to_8bit(image, clip_min=None, clip_max=None):
    """
    Convert a floating-point image to 8-bit (uint8).

    Parameters:
    - image: np.ndarray
        The input image in float format (single or double precision).
    - clip_min: float, optional
        The minimum value for clipping. If None, the minimum of the image is used.
    - clip_max: float, optional
        The maximum value for clipping. If None, the maximum of the image is used.

    Returns:
    - np.ndarray
        The image scaled to 8-bit (0-255 range).
    """
    # Ensure the input is a numpy array
    image = np.array(image, dtype=np.float32)

    # Clip the image to the specified range or the min/max of the image
    clip_min = np.min(image) if clip_min is None else clip_min
    clip_max = np.max(image) if clip_max is None else clip_max
    image = np.clip(image, clip_min, clip_max)

    # Normalize to the range [0, 1]
    normalized = (image - clip_min) / (clip_max - clip_min)

    # Scale to [0, 255] and convert to uint8
    image_8bit = (normalized * 255).astype(np.uint8)

    return image_8bit