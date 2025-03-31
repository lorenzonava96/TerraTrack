import numpy as np
import cv2
from skimage import exposure
from sklearn.preprocessing import StandardScaler
import pandas as pd
from datetime import datetime
from itertools import combinations
import os
import rasterio
import matplotlib.pyplot as plt

def process_composite_image(output_dir, selection_method='auto'):
    """
    Processes a Sentinel-2 composite image by allowing manual or automatic band selection.
    
    Args:
    - output_dir (str): Path to the directory containing the composite image and metadata.
    - selection_method (str): 'auto' to keep all bands from the first stack, 'manual' for user selection.
    
    Saves:
    - A filtered 8-bit composite image with selected bands.
    - Updated metadata CSV indicating kept/removed bands.
    """

    # Define file paths
    composite_path = os.path.join(output_dir, "S2_Composite.tif")
    output_path = os.path.join(output_dir, "S2_Composite_Filtered_8bit.tif")
    metadata_path = os.path.join(output_dir, "S2_Metadata.csv")
    updated_metadata_path = os.path.join(output_dir, "Updated_Metadata.csv")

    # Load metadata
    metadata_df = pd.read_csv(metadata_path)

    # User selection: Auto or Manual
    if selection_method == 'auto':
        band_indices_to_keep = list(range(1, metadata_df.shape[0] + 1))
        kept_or_removed = ['kept'] * len(band_indices_to_keep)
    else:
        # Open composite image
        with rasterio.open(composite_path) as src:
            num_bands = src.count
            print(f"Number of bands in the composite: {num_bands}")

            band_indices_to_keep = []
            kept_or_removed = []

            # Loop through each band for user selection
            for band_index in range(1, num_bands + 1):  # Bands are 1-indexed in rasterio
                band_data = src.read(band_index)

                # Display the band
                plt.figure()
                plt.imshow(band_data, cmap='gray')
                plt.title(f'Band {band_index}')
                plt.colorbar()
                plt.show()

                # Ask user to keep the band
                keep_band = input(f"Keep Band {band_index}? (y/n): ").strip().lower()
                if keep_band == 'y':
                    band_indices_to_keep.append(band_index)
                    kept_or_removed.append('kept')
                else:
                    kept_or_removed.append('removed')

    print(f"Bands to keep: {band_indices_to_keep}")

    # Update metadata
    metadata_df['Status'] = kept_or_removed
    metadata_df.to_csv(updated_metadata_path, index=False)
    print(f"Updated metadata saved to {updated_metadata_path}.")

    # Save selected bands to new composite image
    if band_indices_to_keep:
        with rasterio.open(composite_path) as src:
            meta = src.meta.copy()
            meta.update(count=len(band_indices_to_keep), dtype=rasterio.uint8)

            with rasterio.open(output_path, 'w', **meta) as dst:
                for i, band_index in enumerate(band_indices_to_keep, start=1):
                    band_data = src.read(band_index)

                    # Normalize to 8-bit (0-255)
                    band_data = np.nan_to_num(band_data)
                    band_min, band_max = band_data.min(), band_data.max()
                    band_data_8bit = ((band_data - band_min) / (band_max - band_min) * 255).astype(np.uint8)

                    dst.write(band_data_8bit, i)

        print(f"Filtered composite image saved to {output_path}.")
    else:
        print("No bands selected. Output file not created.")

def preprocess_image_stack(image_stack, preprocess_params):
    """
    Modular preprocessing function for a stack of images using two options: block matching or optical flow.

    Parameters:
    - image_stack: Input stack of images (3D array: (num_bands, height, width)).
    - preprocess_params: Dictionary containing preprocessing options:
        {
            "method": "cross_corr" or "optical_flow",  # Choose preprocessing method
            "custom_preprocessing_func": None  # Optional custom preprocessing function
        }

    Returns:
    - Preprocessed stack of images.
    """
    method = preprocess_params.get("method", "cross_corr")
    custom_preprocessing_func = preprocess_params.get("custom_preprocessing_func", None)

    def naof2(im):
        """Apply NAOF2 anisotropic orientation filtering."""
        # Define the filters
        filter_1 = np.array([-1, 2, -1])
        filter_2 = np.array([[-1], [2], [-1]])
        filter_3 = np.array([[-1, 0, 0], [0, 2, 0], [0, 0, -1]])
        filter_4 = np.array([[0, 0, -1], [0, 2, 0], [-1, 0, 0]])

        # Apply filters using cv2
        filt1 = cv2.filter2D(im, -1, filter_1)
        filt2 = cv2.filter2D(im, -1, filter_2)
        filt3 = cv2.filter2D(im, -1, filter_3)
        filt4 = cv2.filter2D(im, -1, filter_4)

        # Combine filters to recreate the original features
        at1 = np.arctan2(filt1, filt2)
        at2 = np.arctan2(filt3, filt4)

        return np.cos(at1) + np.cos(np.pi / 2 - at1) + np.cos(at2) + np.cos(np.pi / 2 - at2)

    preprocessed_stack = []

    for img in image_stack:
        img_gray = np.nan_to_num(img)  # Replace NaNs with 0

        if method == "cross_corr":
            # Apply NAOF2 filter
            a_temp_f = naof2(img_gray)  # Filter the image
            # print("NAOF2 Output Range:", a_temp_f.min(), a_temp_f.max())  # Debug


            # Rescale intensity to range [0, 1]
            a_temp_f = exposure.rescale_intensity(a_temp_f, out_range=(0, 1))
            # print("Rescaled Intensity Range:", a_temp_f.min(), a_temp_f.max())  # Debug


            # Z-score normalization using StandardScaler
            flat_img = a_temp_f.flatten().reshape(-1, 1)  # Flatten the filtered image
            flat_img = StandardScaler().fit_transform(flat_img)  # Apply Z-score normalization
            img_gray = flat_img.reshape(img_gray.shape)  # Reshape back to the original dimensions

            
        elif method == "optical_flow":
            # Optical Flow preprocessing
            # Convert image to 8-bit if not already
            img_gray = cv2.convertScaleAbs(img_gray, alpha=(255.0/img_gray.max()))

        if custom_preprocessing_func is not None:
            img_gray = custom_preprocessing_func(img_gray)

        preprocessed_stack.append(img_gray)
        
    return np.stack(preprocessed_stack, axis=2)

def define_date_pairs(metadata_path, min_separation=1, max_separation=5):
    """
    Processes date pairs from a metadata CSV and filters them by a specified separation range.

    Parameters:
        metadata_path (str): Path to the metadata CSV file.
        min_separation (float): Minimum separation between dates in years.
        max_separation (float): Maximum separation between dates in years.

    Returns:
        tuple: Filtered date pairs (dat1, dat2) and their separations in years.
    """
    # Load the metadata CSV file
    metadata_df = pd.read_csv(metadata_path)

    # Filter the DataFrame to include only rows where 'Status' is 'kept'
    metadata_df = metadata_df[metadata_df['Status'] == 'kept']
    # print(metadata_df.head())
    # Extract the 'Date' column and convert it to datetime objects
    datax = [datetime.strptime(date, '%Y-%m-%d') for date in metadata_df['Date']]

    # Generate all unique date pairs using itertools.combinations
    dat1, dat2 = zip(*combinations(datax, 2))

    # Convert lists to numpy arrays for easier handling
    dat1 = np.array(dat1)
    dat2 = np.array(dat2)

    # Calculate separations in years (time difference between the dates)
    separation = np.array([(d2 - d1).days / 365.25 for d1, d2 in zip(dat1, dat2)])

    # Filter separations within the specified range
    valid_indices = (separation > min_separation) & (separation < max_separation)
    dat1 = dat1[valid_indices]
    dat2 = dat2[valid_indices]
    separation = separation[valid_indices]

    print(f"Number of valid separations: {len(separation)}")
    return dat1, dat2, separation, datax
