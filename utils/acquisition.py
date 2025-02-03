import ee # type: ignore

def get_evenly_spaced_images_per_year(collection, start_year, end_year, n_per_year):
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
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-30"

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

# Function to compute NDWI and mask water regions
def add_ndwi_and_mask_water(image):
    # Compute NDWI: (Green - NIR) / (Green + NIR)
    ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')

    # Create a mask where NDWI > 0 (water)
    water_mask = ndwi.gt(0)

    # Mask water regions and add NDWI as a band
    masked_image = image.updateMask(water_mask.Not())
    return masked_image.addBands(ndwi)
