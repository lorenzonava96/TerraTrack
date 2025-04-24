import os
import h5py
import numpy as np

def handle_predictions(
    output_dir,
    method,
    match_func,
    results=None,
    separation=None,
    orig=None,
    dat1=None,
    dat2=None,
    load=True
):
    """
    Handles saving/loading prediction results using HDF5 to reduce peak RAM usage.
    If loading succeeds, returns data straight from the file; otherwise builds from
    workspace variables and writes them into an .h5.

    Parameters:
        output_dir (str): Directory where results are saved/loaded.
        method (str): Name of the method (e.g., 'block_matching').
        match_func (str): Name of the matching function.
        results (list): List of dicts with keys 'u','v','feature_points','pkrs','snrs'.
        separation (np.ndarray): Separation vector.
        orig (np.ndarray): Original study area image (3D or 2D stack).
        dat1 (list of str): Start-date strings.
        dat2 (list of str): End-date strings.
        load (bool): If True, try to load before regenerating.

    Returns:
        dict: {
            "all_u": list of np.ndarray,
            "all_v": list of np.ndarray,
            "all_feature_points": list of np.ndarray,
            "all_pkrs": list of np.ndarray,
            "all_snrs": list of np.ndarray,
            "separation": np.ndarray,
            "study_area_image": np.ndarray,
            "dat1": list of str,
            "dat2": list of str
        }
    """
    os.makedirs(output_dir, exist_ok=True)
    h5_path = os.path.join(
        output_dir,
        f"{output_dir}_displacement_results_{method}_{match_func}.h5"
    )

    if load and os.path.exists(h5_path):
        try:
            with h5py.File(h5_path, 'r') as f:
                def _read_list(group_name):
                    grp = f[group_name]
                    # assume keys are '0','1',... as strings
                    return [grp[k][()] for k in sorted(grp, key=lambda x: int(x))]

                data = {
                    "all_u": _read_list('all_u'),
                    "all_v": _read_list('all_v'),
                    "all_feature_points": _read_list('all_feature_points'),
                    "all_pkrs": _read_list('all_pkrs'),
                    "all_snrs": _read_list('all_snrs'),
                    "separation": f['separation'][()],
                    "study_area_image": f['study_area_image'][()],
                    "dat1": [d.decode('utf-8') for d in f['dat1'][()]],
                    "dat2": [d.decode('utf-8') for d in f['dat2'][()]]
                }
            print(f"Loaded data from {h5_path}")
            return data
        except Exception as e:
            print(f"Failed to load HDF5 ({e}), will regenerate.")

    # Build data from workspace variables
    print("Generating data from workspace variables...")

    all_u = [res['u'] for res in results]
    all_v = [res['v'] for res in results]
    all_feature_points = [res['feature_points'] for res in results]
    all_pkrs = [res['pkrs'] for res in results]
    all_snrs = [res['snrs'] for res in results]

    # Flatten study_area_image to 2D if it's a stack
    study_img = orig[..., 0] if orig.ndim == 3 else orig

    # Write out to HDF5
    with h5py.File(h5_path, 'w') as f:
        # helper to write a list of arrays into a subgroup
        def _write_list(name, lst):
            grp = f.create_group(name)
            for idx, arr in enumerate(lst):
                grp.create_dataset(
                    str(idx), data=arr,
                    compression='gzip', chunks=True
                )

        _write_list('all_u', all_u)
        _write_list('all_v', all_v)
        _write_list('all_feature_points', all_feature_points)
        _write_list('all_pkrs', all_pkrs)
        _write_list('all_snrs', all_snrs)

        f.create_dataset(
            'separation', data=separation,
            compression='gzip', chunks=True
        )
        f.create_dataset(
            'study_area_image', data=study_img,
            compression='gzip', chunks=True
        )

        # store dat1/dat2 as UTF-8 variable-length strings
        str_dt = h5py.string_dtype(encoding='utf-8')
        f.create_dataset('dat1', data=np.array(dat1, dtype=str_dt), dtype=str_dt)
        f.create_dataset('dat2', data=np.array(dat2, dtype=str_dt), dtype=str_dt)

    print(f"Saved HDF5 to {h5_path}")
    return {
        "all_u": all_u,
        "all_v": all_v,
        "all_feature_points": all_feature_points,
        "all_pkrs": all_pkrs,
        "all_snrs": all_snrs,
        "separation": separation,
        "study_area_image": study_img,
        "dat1": dat1,
        "dat2": dat2
    }
