import os
import h5py
import numpy as np

def handle_predictions(
    output_dir, acq_id=None, results=None, separation=None, orig=None, dat1=None, dat2=None, save=True, load=True
):
    """
    Handles saving or loading prediction results using HDF5 to reduce memory usage.
    If loading fails, data is created from the provided workspace variables.

    This version stores each element of lists (e.g. all_u, all_v, etc.) in a subgroup,
    and when loading, it reshapes the "all_feature_points" arrays to (-1, 2) if possible.

    Parameters:
        output_dir (str): Directory where results are saved or loaded from.
        acq_id (str): Acquisition ID for naming the file.
        results (list): List of dictionaries containing prediction results.
        separation (np.array): Separation vector.
        orig (np.array): Original study area image.
        dat1 (list): List of start dates.
        dat2 (list): List of end dates.
        save (bool): Whether to save.
        load (bool): Whether to load.

    Returns:
        dict: Prediction data loaded from the file or created from the results.
    """
    file_path = os.path.join(output_dir, f"{acq_id}_displacement_results.h5")

    if load:
        try:
            with h5py.File(file_path, 'r') as hf:
                print(f"Loaded data successfully from {file_path}.")

                def load_group(group_name, reshape=False):
                    grp = hf[group_name]
                    # Load datasets in order assuming keys "0", "1", etc.
                    arr_list = [grp[str(i)][:] for i in range(len(grp))]
                    if reshape:
                        # If each flat array's length is even, reshape it to (-1,2)
                        return [np.reshape(arr, (-1, 2)) if (arr.size % 2 == 0) else arr for arr in arr_list]
                    else:
                        return arr_list

                data = {
                    "all_u": load_group("all_u"),
                    "all_v": load_group("all_v"),
                    "all_feature_points": load_group("all_feature_points", reshape=True),
                    "all_pkrs": load_group("all_pkrs"),
                    "all_snrs": load_group("all_snrs"),
                    "separation": hf["separation"][:],
                    "study_area_image": hf["study_area_image"][:],
                    "dat1": [s.decode('utf-8') if isinstance(s, bytes) else s for s in hf["dat1"][:]],
                    "dat2": [s.decode('utf-8') if isinstance(s, bytes) else s for s in hf["dat2"][:]]
                }
                return data
        except FileNotFoundError:
            print(f"File not found: {file_path}. Switching to data creation from workspace.")
            load = False

    if not load:
        # Validate required variables.
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

        # Initialize lists to store results.
        all_u, all_v, all_feature_points = [], [], []
        all_pkrs, all_snrs = [], []

        for result in results:
            all_u.append(result['u'])
            all_v.append(result['v'])
            all_feature_points.append(result['feature_points'])
            all_pkrs.append(result['pkrs'])
            all_snrs.append(result['snrs'])

        os.makedirs(output_dir, exist_ok=True)
        with h5py.File(file_path, 'w') as hf:
            # Helper function: create a subgroup and store each element as a dataset.
            def save_list(group_name, data_list):
                grp = hf.create_group(group_name)
                for i, arr in enumerate(data_list):
                    # Store as a flat array of float32.
                    grp.create_dataset(str(i), data=np.array(arr, dtype=np.float32).ravel(), compression="gzip")
            
            save_list("all_u", all_u)
            save_list("all_v", all_v)
            save_list("all_feature_points", all_feature_points)
            save_list("all_pkrs", all_pkrs)
            save_list("all_snrs", all_snrs)
            
            # Save fixed-shape arrays.
            hf.create_dataset("separation", data=separation, compression="gzip")
            hf.create_dataset("study_area_image", data=orig[..., 0], compression="gzip")
            # Save dates as fixed-length strings.
            hf.create_dataset("dat1", data=np.array(dat1, dtype='S'), compression="gzip")
            hf.create_dataset("dat2", data=np.array(dat2, dtype='S'), compression="gzip")

            print(f"Results saved in HDF5 file: {file_path}")

        data = {
            "all_u": all_u,
            "all_v": all_v,
            "all_feature_points": all_feature_points,
            "all_pkrs": all_pkrs,
            "all_snrs": all_snrs,
            "separation": separation,
            "study_area_image": orig[..., 0],
            "dat1": dat1,
            "dat2": dat2
        }
        return data
