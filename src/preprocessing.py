import ee # type: ignore
import os
import pandas as pd
import geemap

def get_sentinel2_collection(roi, cloud_cover_max=10):
    """
    Fetch Sentinel-2 Harmonized collection with cloud filtering.
    """
    return ee.ImageCollection('COPERNICUS/S2_HARMONIZED') \
        .filterBounds(roi) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover_max)) \
        .select(['B4', 'B3', 'B2', 'B8', 'B11', 'QA60'])

def check_full_coverage(image, roi, error_margin=10):
    """
    Check if an image fully contains the ROI.
    """
    contains_roi = image.geometry().contains(roi, ee.ErrorMargin(error_margin))
    return image.set('contains_roi', contains_roi)

def filter_full_coverage(collection, roi):
    """
    Filter images that fully cover the ROI.
    """
    collection_with_full_coverage = collection.map(lambda img: check_full_coverage(img, roi))
    return collection_with_full_coverage.filter(ee.Filter.eq('contains_roi', True))

def add_ndwi_and_mask_water(image):
    """
    Computes NDWI and masks water regions.
    """
    ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
    water_mask = ndwi.gt(0)
    masked_image = image.updateMask(water_mask.Not())
    return masked_image.addBands(ndwi)

def load_dem_and_morpho(roi):
    """
    Loads DEM and calculates slope & aspect.
    """
    dem = ee.Image('USGS/SRTMGL1_003').clip(roi)
    slope = ee.Terrain.slope(dem)
    aspect = ee.Terrain.aspect(dem)
    return dem.addBands(slope.rename('slope')).addBands(aspect.rename('aspect'))

def export_s2_composite_morpho_and_metadata(
    final_collection,
    morpho,
    roi,
    output_dir,
    band="B8",
    composite_name="S2_Composite.tif",
    morpho_name="morpho.tif",
    metadata_name="S2_Metadata.csv",
    scale=None,
    crs=None,
    quiet=False,
):
    """
    Exports:
      1) Sentinel-2 composite by stacking `band` across `final_collection` (toBands())
      2) `morpho` ee.Image (e.g., DEM+slope+aspect stacked)
      3) CSV metadata (Image_ID, Date, Cloud_Cover, Orbit_Number, Tile_ID)

    Notes:
      - If scale/crs are None, EE/geemap will pick defaults (less deterministic).
      - roi.getInfo() is evaluated once and reused.
    """
    os.makedirs(output_dir, exist_ok=True)
    region = roi.getInfo()

    # --- Composite (stack band over time to bands) ---
    composite_image = final_collection.select(band).toBands()
    composite_path = os.path.join(output_dir, composite_name)
    geemap.download_ee_image(
        composite_image,
        composite_path,
        scale=scale,
        crs=crs,
        region=region,
    )
    if not quiet:
        print(f"Composite image downloaded: {composite_path}")

    # --- Morpho ---
    morpho_path = os.path.join(output_dir, morpho_name)
    geemap.download_ee_image(
        morpho,
        morpho_path,
        scale=scale,
        crs=crs,
        region=region,
    )
    if not quiet:
        print(f"DEM, Slope, and Aspect downloaded: {morpho_path}")

    # --- Metadata ---
    # (Each getInfo triggers a request; kept readable. Can be optimized further if needed.)
    image_ids = final_collection.aggregate_array("system:index").getInfo()

    # Convert timestamps to YYYY-MM-dd in EE, then bring to client
    dates = (
        final_collection
        .aggregate_array("system:time_start")
        .map(lambda t: ee.Date(t).format("YYYY-MM-dd"))
        .getInfo()
    )

    cloud_covers = final_collection.aggregate_array("CLOUDY_PIXEL_PERCENTAGE").getInfo()
    orbits = final_collection.aggregate_array("SENSING_ORBIT_NUMBER").getInfo()
    tile_ids = final_collection.aggregate_array("MGRS_TILE").getInfo()

    metadata_df = pd.DataFrame({
        "Image_ID": image_ids,
        "Date": dates,
        "Cloud_Cover": cloud_covers,
        "Orbit_Number": orbits,
        "Tile_ID": tile_ids,
    })

    # Clean ID column (keep trailing timestamp-like token)
    metadata_df["Image_ID"] = metadata_df["Image_ID"].str.replace(
        r"^.*?(\d{8}T\d{6}.*)$", r"\1", regex=True
    )

    metadata_path = os.path.join(output_dir, metadata_name)
    metadata_df.to_csv(metadata_path, index=False)
    if not quiet:
        print(f"Metadata saved to CSV file: {metadata_path}")

    return {
        "composite_path": composite_path,
        "morpho_path": morpho_path,
        "metadata_path": metadata_path,
        "metadata_df": metadata_df,  # handy for immediate inspection
    }

def process_sentinel2_data(roi, start_year, end_year, SUMMER_START, SUMMER_END, 
                           cloud_cover_max=10, n_per_year=4, mask_water=False, 
                           check_clouds=True, cloud_threshold=5, check_snow=False, snow_threshold=5,
                           start_date=None, final_date=None):

    """
    Processes Sentinel-2 data with optional water masking and removes duplicates, 
    images with too many clouds, and optionally images with too much snow in the ROI.
    Also removes images after a specified final date.

    Parameters:
    - roi: Region of interest (Earth Engine Geometry)
    - start_year: Start year for data collection
    - end_year: End year for data collection
    - SUMMER_START: Start of the summer season (month-day format, e.g., '06-01')
    - SUMMER_END: End of the summer season (month-day format, e.g., '08-31')
    - cloud_cover_max: Maximum cloud cover percentage per tile (default: 10)
    - n_per_year: Number of evenly spaced images per year (default: 4)
    - mask_water: Apply water masking (default: False)
    - check_clouds: Remove images with too many clouds in ROI (default: True)
    - cloud_threshold: Maximum allowed cloud percentage inside ROI (default: 5%)
    - check_snow: Remove images with too much snow in ROI (default: False)
    - snow_threshold: Maximum allowed snow percentage inside ROI (default: 5%)
    - final_date: Last date to include images (format: 'YYYY-MM-DD', default: None)

    Returns:
    - final_collection: Sentinel-2 image collection (filtered for clouds, snow, and duplicates)
    - morpho: DEM and morphometry data
    """

    roi_expanded = roi.buffer(1)

    # Fetch and filter Sentinel-2 collection (tile-wide cloud filtering)
    collection = get_sentinel2_collection(roi_expanded, cloud_cover_max)
    print(f"Number of images after date and tile-wide cloud filtering: {collection.size().getInfo()}")

    # Filter full coverage images
    filtered_collection = filter_full_coverage(collection, roi)
    print(f"Number of images fully covering the ROI: {filtered_collection.size().getInfo()}")

    # Remove images where clouds cover more than `cloud_threshold%` of the ROI
    if check_clouds:
        def filter_cloudy_images(image):
            cloud_percentage = get_cloud_percentage(image, roi)
            return image.set('cloud_percentage', cloud_percentage)

        # Apply the function to calculate cloud percentage
        filtered_collection = filtered_collection.map(filter_cloudy_images)

        # Remove images where cloud_percentage is missing (null)
        filtered_collection = filtered_collection.filter(ee.Filter.notNull(['cloud_percentage']))

        # Remove images with clouds above threshold
        filtered_collection = filtered_collection.filter(ee.Filter.lte('cloud_percentage', cloud_threshold))

        print(f"Number of images after ROI-based cloud filtering: {filtered_collection.size().getInfo()}")

    # Remove images where snow covers more than `snow_threshold%` of the ROI
    if check_snow:
        def filter_snowy_images(image):
            snow_percentage = get_snow_percentage(image, roi)
            return image.set('snow_percentage', snow_percentage)

        # Apply the function to calculate snow percentage
        filtered_collection = filtered_collection.map(filter_snowy_images)

        # Remove images where snow_percentage is missing (null)
        filtered_collection = filtered_collection.filter(ee.Filter.notNull(['snow_percentage']))

        # Remove images with snow above threshold
        filtered_collection = filtered_collection.filter(ee.Filter.lte('snow_percentage', snow_threshold))

        print(f"Number of images after ROI-based snow filtering: {filtered_collection.size().getInfo()}")

    # Remove images before the specified start_date
    if start_date:
        filtered_collection = filtered_collection.filter(ee.Filter.date(start_date, '2100-01-01'))
        print(f"Number of images after applying start_date filter ({start_date}): {filtered_collection.size().getInfo()}")

    # Remove images after the specified final_date
    if final_date:
        filtered_collection = filtered_collection.filter(ee.Filter.date('1970-01-01', final_date))
        print(f"Number of images after applying final_date filter ({final_date}): {filtered_collection.size().getInfo()}")

    # Select evenly spaced images per year (and within summer period)
    final_collection = get_evenly_spaced_images_per_year(filtered_collection, start_year, end_year, n_per_year, SUMMER_START, SUMMER_END)

    # Conditionally apply NDWI and water masking
    if mask_water:
        final_collection = final_collection.map(add_ndwi_and_mask_water)

    # Remove duplicate tiles by keeping only the first occurrence of each MGRS_TILE per date
    def set_date_tile_property(image):
        date = ee.Date(image.get('system:time_start')).format('YYYY-MM-DD')
        return image.set('date_tile', ee.String(date).cat('_').cat(image.get('MGRS_TILE')))

    final_collection = final_collection.map(set_date_tile_property)

    # Apply distinct() and convert back to ImageCollection
    distinct_images = final_collection.distinct(['date_tile'])
    final_collection = ee.ImageCollection(distinct_images.map(lambda f: ee.Image(f)))

    print(f"Number of images after removing duplicate tiles: {final_collection.size().getInfo()}")

    # Load DEM and morphometry
    morpho = load_dem_and_morpho(roi)

    return final_collection, morpho


def get_cloud_percentage(image, roi):
    """
    Computes the percentage of clouds inside the ROI using the QA60 cloud mask.
    
    Parameters:
    - image: Sentinel-2 image (Earth Engine Image)
    - roi: Region of Interest (Earth Engine Geometry)

    Returns:
    - cloud_percentage (ee.Number): Percentage of clouds within the ROI (returns 0 if no data)
    """

    # Extract the QA60 cloud mask band (1 = cloud, 0 = clear)
    cloud_mask = image.select('QA60').gt(0)

    # Compute the total number of pixels in the ROI
    total_pixels = cloud_mask.reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=roi,
        scale=30,
        bestEffort=True
    ).get('QA60')

    # Compute the number of cloudy pixels in the ROI
    cloud_pixels = cloud_mask.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=roi,
        scale=30,
        bestEffort=True
    ).get('QA60')

    # Handle cases where no valid pixels exist
    total_pixels = ee.Algorithms.If(ee.Algorithms.IsEqual(total_pixels, None), ee.Number(1), total_pixels)
    cloud_pixels = ee.Algorithms.If(ee.Algorithms.IsEqual(cloud_pixels, None), ee.Number(0), cloud_pixels)

    # Compute cloud percentage: (cloud pixels / total pixels) * 100
    cloud_percentage = ee.Number(cloud_pixels).divide(ee.Number(total_pixels)).multiply(100)

    return cloud_percentage

def get_snow_percentage(image, roi):
    """
    Computes the percentage of snow-covered area inside the ROI using the NDSI (Normalized Difference Snow Index).
    
    Parameters:
    - image: Sentinel-2 image (Earth Engine Image)
    - roi: Region of Interest (Earth Engine Geometry)

    Returns:
    - snow_percentage (ee.Number): Percentage of snow within the ROI (returns 0 if no data)
    """

    # Compute the Normalized Difference Snow Index (NDSI)
    # NDSI = (Green - SWIR) / (Green + SWIR)
    ndsi = image.normalizedDifference(['B3', 'B11']).rename('NDSI')

    # Define the snow threshold (commonly used threshold: NDSI > 0.4)
    snow_mask = ndsi.gt(0.4)

    # Compute the total number of pixels in the ROI
    total_pixels = snow_mask.reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=roi,
        scale=30,
        bestEffort=True
    ).get('NDSI')

    # Compute the number of snow-covered pixels in the ROI
    snow_pixels = snow_mask.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=roi,
        scale=30,
        bestEffort=True
    ).get('NDSI')

    # Handle cases where no valid pixels exist
    total_pixels = ee.Algorithms.If(ee.Algorithms.IsEqual(total_pixels, None), ee.Number(1), total_pixels)
    snow_pixels = ee.Algorithms.If(ee.Algorithms.IsEqual(snow_pixels, None), ee.Number(0), snow_pixels)

    # Compute snow percentage: (snow pixels / total pixels) * 100
    snow_percentage = ee.Number(snow_pixels).divide(ee.Number(total_pixels)).multiply(100)

    return snow_percentage

def get_evenly_spaced_images_per_year(collection, start_year, end_year, n_per_year, SUMMER_START, SUMMER_END):
    """
    Filters images per year, ensuring at least one image if available.
    - Skips years with no images.
    - Keeps 1 image if only 1 exists.
    - Selects exactly `N_PER_YEAR` evenly spaced images when more exist.
    """
    final_collection = ee.ImageCollection([])

    # Detect the first available year
    first_image = collection.sort('system:time_start').first()
    if first_image:
        first_available_year = ee.Date(first_image.get('system:time_start')).get('year').getInfo()
        print(f"Adjusting START_YEAR to first available data year: {first_available_year}")
        start_year = max(start_year, first_available_year)
    else:
        print("No images found at all. Check filtering criteria.")
        return ee.ImageCollection([])

    for year in range(start_year, end_year + 1):
        start_date = f"{year}{SUMMER_START}"
        end_date = f"{year}{SUMMER_END}"

        # Filter images for this year
        year_images = collection.filterDate(start_date, end_date)
        total_images = year_images.size().getInfo()  # Convert to Python integer

        # Debugging: Print available images per year
        print(f"Year {year}: {total_images} images available")

        # Skip years with no images
        if total_images == 0:
            print(f"Skipping {year}: No images found")
            continue

        # If there is only 1 image, keep it
        if total_images == 1:
            print(f"  → Only 1 image available. Keeping it.")
            final_collection = final_collection.merge(year_images)
            continue

        # If more than N_PER_YEAR images exist, evenly distribute selection
        if total_images > n_per_year:
            image_list = year_images.toList(total_images)

            # Compute step size for even distribution
            step = total_images // n_per_year  # Ensures step is always ≥ 1
            indices = ee.List.sequence(0, total_images - 1, step)

            # Select exactly N_PER_YEAR images
            indices = indices.slice(0, n_per_year)

            # Select images at those indices
            sampled_images = indices.map(lambda i: image_list.get(ee.Number(i).int()))

            # Convert back to ImageCollection
            year_images = ee.ImageCollection.fromImages(sampled_images)

        # Merge into final collection
        final_collection = final_collection.merge(year_images)

    return final_collection
