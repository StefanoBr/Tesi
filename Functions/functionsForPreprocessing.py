import rasterio
from rasterio.mask import mask
import geopandas as gpd
import numpy as np
import os
import glob
from shapely.geometry import box
import matplotlib.pyplot as plt
import sys


def process_raster(aoi, output_dir, raster_path):
    """Clip and compute cloud coverage for a single raster."""
    #print(f"Processing {raster_path} with output directory {output_dir}")
    try:
        print(f"Opening raster: {raster_path}")
        # Open the raster
        with rasterio.open(raster_path) as src:
            print(f"Checking CRS compatibility: {src.crs} vs {aoi.crs}")
            # Ensure CRS compatibility
            
            if src.crs.to_string() != aoi.crs.to_string():
                print("Reprojecting AOI to match raster CRS...")
                aoi = aoi.to_crs(src.crs.to_string())

            print("Clipping raster...")
            # Clip raster
            clipped, clipped_transform = mask(src, aoi.geometry, crop=True, nodata=src.nodata)
            print("Computing cloud coverage...")
            # Compute cloud coverage (using bit 3 for clouds)
            cloud_mask = np.bitwise_and(clipped, 1 << 3).astype(bool)
            cloud_coverage = np.sum(cloud_mask) / np.count_nonzero(clipped != src.nodata) * 100

            # Save clipped raster if cloud coverage is acceptable
            if cloud_coverage < 3.0:
                print("Saving clipped raster...")
                output_file = os.path.join(output_dir, os.path.basename(raster_path))
                meta = src.meta.copy()
                meta.update({"driver": "GTiff", "height": clipped.shape[1], "width": clipped.shape[2], "transform": clipped_transform})
                with rasterio.open(output_file, "w", **meta) as dst:
                    dst.write(clipped)

            return os.path.basename(raster_path), cloud_coverage

    except Exception as e:
        print(f"Error processing {raster_path}: {e}")
        return os.path.basename(raster_path), None

def clip_rasters_by_aoi(input_folder, output_folder, file_suffix, aoi_shapefile):
    """
    Clips all raster files ending with a specific suffix in the input folder 
    based on an AOI shapefile and saves the clipped rasters in the output folder.

    Parameters:
        input_folder (str): Path to the folder containing input raster files.
        output_folder (str): Path to the folder where clipped rasters will be saved.
        file_suffix (str): File suffix to filter raster files (e.g., 'ST_B10.TIF').
        aoi_shapefile (str): Path to the shapefile defining the AOI.
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Read AOI shapefile
    aoi = gpd.read_file(aoi_shapefile)
    if aoi.crs is None:
        raise ValueError("The AOI shapefile must have a CRS defined.")

    # Find all raster files ending with the given suffix
    raster_files = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.endswith(file_suffix)
    ]

    if not raster_files:
        print(f"No files ending with '{file_suffix}' found in {input_folder}.")
        return

    print(f"Found {len(raster_files)} files to clip.")
    
    # Process each raster file
    for raster_path in raster_files:
        try:
            print(f"Processing: {raster_path}")
            with rasterio.open(raster_path) as src:
                # Ensure AOI CRS matches raster CRS
                if src.crs != aoi.crs:
                    aoi = aoi.to_crs(src.crs)

                # Clip the raster using the AOI geometry
                clipped, clipped_transform = mask(
                    src, aoi.geometry, crop=True, nodata=src.nodata
                )

                # Prepare metadata for the output raster
                output_meta = src.meta.copy()
                output_meta.update({
                    "driver": "GTiff",
                    "height": clipped.shape[1],
                    "width": clipped.shape[2],
                    "transform": clipped_transform,
                    "nodata": src.nodata,
                })

                # Generate output file path
                output_file = os.path.join(output_folder, os.path.basename(raster_path))

                # Save the clipped raster
                with rasterio.open(output_file, "w", **output_meta) as dst:
                    dst.write(clipped)

                print(f"Clipped raster saved to: {output_file}")

        except Exception as e:
            print(f"Error processing {raster_path}: {e}")

def compute_overlapping_percentage_with_mask(output_dir, aoi, threshold_1=55):

    # Retrieve raster files in the output directory
    raster_files = [
        file for file in glob.glob(os.path.join(output_dir, "*.TIF"))
        if os.path.isfile(file)
    ]

    count_above_threshold_1 = 0
    files_above_threshold = []  # To store file paths exceeding the threshold

    for raster_path in raster_files:
        #print (raster_path)
        try:
            with rasterio.open(raster_path) as src:
                # Ensure the raster has a CRS
                if src.crs is None:
                    raise ValueError(f"Raster {raster_path} does not have a CRS.")

                # Create a bounding box for the raster
                raster_bounds_polygon = box(*src.bounds)
                raster_bounds = gpd.GeoSeries([raster_bounds_polygon], crs=src.crs)
                #print(f"Raster CRS: {raster_bounds.crs}")

                # Ensure AOI CRS matches raster CRS
                if aoi.crs != raster_bounds.crs:
                    aoi_reprojected = aoi.to_crs(raster_bounds.crs)
                else:
                    aoi_reprojected = aoi
                
                #print(f"AOI CRS: {aoi_reprojected.crs}")
                # Read the raster data
                raster_data = src.read(1)  # Read the first band
                #print("First band values: ", raster_data)
                no_data_value = src.nodata
                raster_mask = np.ones_like(raster_data, dtype=bool)
                #print("\nStep 2a: Initial Raster Mask (All True)")
                #print(raster_mask.astype(int))  # Convert to int for better readability
                # Exclude no-data and specific value areas
                if no_data_value is not None:
                    raster_mask = raster_data != no_data_value
                #print("\nStep 2b: Raster Mask after Excluding No-Data Pixels")
                #print(raster_mask.astype(int))

                valid_data_mask = raster_data > 1  # Adjust this condition if necessary

                # Combine masks
                final_mask = raster_mask & valid_data_mask
                #print(f"Valid data pixel count: {np.sum(final_mask)}")
                

                # Calculate the valid data area
                pixel_width = abs(src.transform[0])  # X-axis pixel size
                pixel_height = abs(src.transform[4])  # Y-axis pixel size
                #print(f"Pixel width: {pixel_width}, Pixel height: {pixel_height}")
                pixel_area = pixel_width * pixel_height  # Area of one pixel
                true_pixel_count = np.sum(final_mask)
                valid_data_area = true_pixel_count * pixel_area
                #print(f"Valid data area: {valid_data_area}")
                
                # Calculate intersection area with AOI
                #intersection = raster_bounds.intersection(aoi_reprojected.geometry.unary_union)
                #intersection_area = intersection.area.sum()
                intersection_area = aoi_reprojected.area.sum()
                #print(f"Intersection area: {intersection_area}")
                #print(f"Raster bounds: {raster_bounds}")
                #print(f"AOI bounds: {aoi_reprojected.bounds}")
                #print(aoi_reprojected.intersects(raster_bounds.unary_union))


                if valid_data_area > 0:
                    overlap_percentage = (valid_data_area / intersection_area) * 100
                    #print("overlap percentage:", overlap_percentage)
                else:
                    overlap_percentage = 0

                # Check thresholds
                if overlap_percentage > threshold_1:
                    count_above_threshold_1 += 1
                    files_above_threshold.append(raster_path)  # Add to the list


        except Exception as e:
            print(f"Error processing {raster_path}: {e}")

    # Plotting --------------------------------------------------------------------------------------------
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    plt.tight_layout(pad=5.0)  # Add padding between subplots

    # Step 1: Plot raster data
    img1 = axs[0, 0].imshow(raster_data, cmap='viridis')
    axs[0, 0].set_title("Step 1: Raster Data")
    plt.colorbar(img1, ax=axs[0, 0], orientation='vertical', label='Pixel Value Range')

    # Step 2: Plot no-data mask
    img2 = axs[0, 1].imshow(raster_mask, cmap='gray')
    axs[0, 1].set_title("Step 2: No-Data Mask")
    plt.colorbar(img2, ax=axs[0, 1], orientation='vertical', label='True/False Mask')

    # Step 3: Plot valid data mask
    img3 = axs[1, 0].imshow(valid_data_mask, cmap='gray')
    axs[1, 0].set_title("Step 3: Valid Data Mask (Value > 1)")
    plt.colorbar(img3, ax=axs[1, 0], orientation='vertical', label='True/False Mask')

    # Step 4: Plot final mask
    img4 = axs[1, 1].imshow(final_mask, cmap='gray')
    axs[1, 1].set_title("Step 4: Final Mask")
    plt.colorbar(img4, ax=axs[1, 1], orientation='vertical', label='True/False Mask')

    # Show the plots
    plt.show()
    print(f"Files with >{threshold_1}% overlap: {count_above_threshold_1}")
    return count_above_threshold_1, files_above_threshold

def process_temperature_files(st_b10_files, st_qa_files, output_folder, uncertaintyThreshold):
    """
    Process ST_B10 and ST_QA files to apply a mask based on uncertainty in ST_QA files
    and convert ST_B10 values to Kelvin.

    Parameters:
        st_b10_files (list): List of file paths for ST_B10 files.
        st_qa_files (list): List of file paths for ST_QA files.
        output_folder (str): Folder to save the processed ST_B10 files.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Ensure both lists have the same length
    if len(st_b10_files) != len(st_qa_files):
        raise ValueError("The number of ST_B10 and ST_QA files must be the same.")

    # Create dictionaries to map prefix to file path
    b10_dict = {os.path.basename(f).rsplit("_ST_B10.TIF", 1)[0]: f for f in st_b10_files}
    qa_dict = {os.path.basename(f).rsplit("_ST_QA.TIF", 1)[0]: f for f in st_qa_files}

    # Match files based on their prefix
    matched_files = []
    for prefix, b10_path in b10_dict.items():
        if prefix in qa_dict:
            matched_files.append((b10_path, qa_dict[prefix])) # Tuple containing band uset to build the mask and band that need to be masked
    print(matched_files)
    print("Matching files found:", len(matched_files))

    # Process each pair of files
    for st_b10_path, st_qa_path in matched_files:
        try:
            # Open ST_B10 file
            with rasterio.open(st_b10_path) as b10_src:
                st_b10_data = b10_src.read(1)  # Read the first band
                b10_meta = b10_src.meta.copy()  # Copy metadata

            # Open ST_QA file
            with rasterio.open(st_qa_path) as qa_src:
                st_qa_data = qa_src.read(1)  # Read the first band
                if qa_src.meta["transform"] != b10_meta["transform"]:
                    raise ValueError("The ST_QA and ST_B10 files do not align spatially.")

            # Apply the scale factor for ST_QA (0.01)
            uncertainty_kelvin = st_qa_data * 0.01

            # Create a mask for uncertainty > 3 Kelvin
            mask = uncertainty_kelvin > uncertaintyThreshold

            # Apply the mask to ST_B10 (set masked values to NaN), no that it is not necessary to transform the ST_B10 data to Kelvin before applying the mask. The masking process is based on the ST_QA uncertainty band, which is already scaled and independent of the temperature unit of the ST_B10 band
            st_b10_data = np.where(mask, np.nan, st_b10_data)

            # Convert ST_B10 values to Kelvin using scale factor and offset
            scale_factor = 0.00341802
            offset = 149
            st_b10_data = st_b10_data * scale_factor + offset

            # Save the processed ST_B10 file
            output_file = os.path.join(output_folder, os.path.basename(st_b10_path))
            b10_meta.update({
                "dtype": "float32",  # Update data type for float values
                "nodata": np.nan,    # Set nodata to NaN
            })
            with rasterio.open(output_file, "w", **b10_meta) as dst:
                dst.write(st_b10_data, 1)

            #print(f"Processed and saved: {output_file}")

        except Exception as e:
            print(f"Error processing {st_b10_path} and {st_qa_path}: {e}")

def create_large_indexed_netcdf2(input_folder, output_file, aoi_bbox, resolution=30, batch_size=10):
    """
    Create an indexed NetCDF from all .TIF files in a folder.

    Parameters:
    - input_folder: Path to the folder containing Landsat .TIF files.
    - output_file: Path to the output NetCDF file.
    - batch_size: Number of files to process in each batch.
    """
    #batch_size=12 # IMPOSTARE VALORE DIVISIBILE SENZA RESTO
    import time
    from tqdm import tqdm
    from datetime import datetime
    from rasterio.transform import from_origin
    import numpy as np
    import pandas as pd
    import xarray as xr
    import rioxarray
    import re

    width = int((aoi_bbox["max_lon"] - aoi_bbox["min_lon"]) / resolution)
    height = int((aoi_bbox["max_lat"] - aoi_bbox["min_lat"]) / resolution)
    x_coords = np.linspace(aoi_bbox["min_lon"], aoi_bbox["max_lon"], width)
    y_coords = np.linspace(aoi_bbox["max_lat"], aoi_bbox["min_lat"], height)  # Flip for descending y
    reference_raster = xr.DataArray(
        np.full((height, width), np.nan),
        dims=("y", "x"),
        coords={"y": y_coords, "x": x_coords},
        attrs={"crs": "EPSG:32632"},  # CRS metadata
    )
    print(f"Reference raster created with dimensions: {height}x{width} (y x x).")

    # Ensure output directory exists otherwise it will be created
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Writing NetCDF file to: {output_file}")

    # Gather all .TIF files
    filenames = sorted([f for f in os.listdir(input_folder) if f.endswith(".TIF")])
    total_files = len(filenames)
    print(f"Found {total_files} .TIF files to process.")

    # Initialize progress bar for batches
    with tqdm(total=total_files, desc="Processing Files", unit="file") as pbar:
        for i in range(0, total_files, batch_size):
            batch_files = filenames[i:i + batch_size]
            #print(f"Processing batch {i // batch_size + 1}/{(total_files // batch_size) + 1}...")
            print(f"\nProcessing batch {i // batch_size + 1} with {len(batch_files)} files...")
            batch_data = []
            batch_time_index = []
            skipped_files = 0

            # Process each file in the current batch
            for filename in batch_files:
                file_path = os.path.join(input_folder, filename)
                try:# Extract acquisition date from the filename
                    match = re.search(r"^.*?_.*?_.*?_(\d{8})_", filename)
                    if match:
                        date_str = match.group(1)
                        date = datetime.strptime(date_str, "%Y%m%d")
                    else:
                        skipped_files += 1
                        print(f"Warning: Could not extract date from {filename}. Skipping.")
                        pbar.update(1)  # Increment progress for skipped file
                        continue
                    #print(f"Skipped {skipped_files} files in batch {i // batch_size + 1}.")
                    # Load raster file
                    #try:
                    raster = rioxarray.open_rasterio(file_path, masked=True, chunks={"x": 1000, "y": 1000})
                    if raster.rio.crs != "EPSG:32632":
                        print(f"Reprojecting raster: {filename}")
                        raster = raster.rio.reproject("EPSG:32632")
                    raster = raster.squeeze().rename({"x": "x", "y": "y"})
                    #raster_padded = raster.combine_first(reference_raster)
                    raster_padded = raster.reindex_like(reference_raster, method="nearest", fill_value=np.nan)
                    # Align raster to reference grid
                    #raster = raster.reindex_like(reference_raster, method="nearest", fill_value=np.nan)
                    batch_data.append(raster_padded)
                    
                    #date_str = re.search(r"_(\d{8})_", filename).group(1)
                    batch_time_index.append(date)
                    #except Exception as e:
                        #print(f"Error loading {filename}: {e}")
                    pbar.update(1)
                        #continue

                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    pbar.update(1)
                    continue
            # Combine batch into a dataset
            if batch_data:
                print(f"Batch {i // batch_size + 1} has {len(batch_data)} rasters and {len(batch_time_index)} time indices.")
                if len(batch_time_index) != len(batch_data):
                    raise ValueError("Mismatch between time index and raster data in this batch.")
                
                batch_stack = xr.concat(batch_data, dim="time")
                #batch_stack["time"] = batch_time_index

                batch_stack["time"] = ("time", batch_time_index)  # Explicitly define the time dimension
                batch_stack["time"] = pd.to_datetime(batch_stack["time"])
                batch_stack.name = "LST"
                
                print(batch_stack["time"])

                # Save batch to NetCDF
                #encoding = {"LST": {"zlib": True, "complevel": 5}}
                encoding = {"LST": {"zlib": False}}
                mode = 'w' if i == 0 else 'a'  # Write for first batch, append for others
                # Check dimensions before saving

                if os.path.exists(output_file):
                    existing_ds = xr.open_dataset(output_file)
                    print(f"Existing NetCDF dimensions: {existing_ds.dims}")
                    existing_ds.close()
                
                print(f"Saving batch {i // batch_size + 1} to NetCDF...")

                start_time = time.time()
                batch_stack.to_netcdf(output_file, mode=mode, encoding=encoding, engine="netcdf4") #ADDED
                #batch_stack.to_netcdf(output_file, mode=mode, encoding=encoding, engine="netcdf4", unlimited_dims=["time"],)
                end_time = time.time()
                #print(f"Batch {i // batch_size + 1} saved in {end_time - start_time:.2f} seconds.")
                #print(f"Batch {i // batch_size + 1} processed successfully.")
            #print(f"Batch {i // batch_size + 1} processed. Skipped {skipped_files} files.")
    print(f"NetCDF file saved at: {output_file}")

def update_netcdf(input_folder, output_file, aoi_bbox, resolution=30, batch_size=1):
    """
    Create an indexed NetCDF from all .TIF files in a folder.

    Parameters:
    - input_folder: Path to the folder containing Landsat .TIF files.
    - output_file: Path to the output NetCDF file.
    - batch_size: Number of files to process in each batch.
    """
    #batch_size=12 # IMPOSTARE VALORE DIVISIBILE SENZA RESTO
    import time
    from tqdm import tqdm
    from datetime import datetime
    from rasterio.transform import from_origin
    import numpy as np
    import pandas as pd
    import xarray as xr
    import rioxarray
    import re

     # Gather all .TIF files
    filenames = sorted([f for f in os.listdir(input_folder) if f.endswith(".TIF")])
    if not filenames:
        raise ValueError("No .TIF files found in the input folder.")
    
    print(f"Found {len(filenames)} .TIF files.")
     # Use the first .TIF file to extract reference grid information
    first_file = os.path.join(input_folder, filenames[0])
    with rasterio.open(first_file) as src:
        bounds = src.bounds  # Get the spatial extent
        resolution_x, resolution_y = src.res  # Get pixel size (resolution)
        width, height = src.width, src.height  # Get dimensions
        crs = src.crs  # Get CRS

    print(f"Reference raster info from: {first_file}")
    print(f"Bounds: {bounds}")
    print(f"Resolution: {resolution_x}, {resolution_y}")
    print(f"Dimensions: {width} x {height}")
    print(f"CRS: {crs}")
    
    # Reproject bounds to EPSG:4326
    with rasterio.Env():
        from pyproj import Transformer
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        min_lon, min_lat = transformer.transform(bounds.left, bounds.bottom)
        max_lon, max_lat = transformer.transform(bounds.right, bounds.top)
    
    # Create reference raster in EPSG:4326
    x_coords = np.linspace(min_lon, max_lon, width)
    y_coords = np.linspace(max_lat, min_lat, height)  # Flip for descending y
    reference_raster = xr.DataArray(
        np.full((height, width), np.nan),
        dims=("y", "x"),
        coords={"y": y_coords, "x": x_coords},
        attrs={"crs": "EPSG:4326"},  # Set CRS metadata
    )

    # x_coords = np.linspace(bounds.left, bounds.right, width)
    # y_coords = np.linspace(bounds.top, bounds.bottom, height)  # Flip for descending y
    # reference_raster = xr.DataArray(
    #     np.full((height, width), np.nan),
    #     dims=("y", "x"),
    #     coords={"y": y_coords, "x": x_coords},
    #     attrs={"crs": crs.to_string()},  # Use the CRS from the .TIF file
    # )
    
    print(f"Reference raster grid created with dimensions: {height} x {width} in EPSG:4326.")

    # Ensure output directory exists otherwise it will be created
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Writing NetCDF file to: {output_file}")
    
    total_files = len(filenames)

    # Initialize progress bar for batches
    with tqdm(total=total_files, desc="Processing Files", unit="file") as pbar:
        for i in range(0, total_files, batch_size):
            batch_files = filenames[i:i + batch_size]
            #print(f"Processing batch {i // batch_size + 1}/{(total_files // batch_size) + 1}...")
            #print(f"\nProcessing batch {i // batch_size + 1} with {len(batch_files)} files...")
            batch_data = []
            batch_time_index = []
            skipped_files = 0

            # Process each file in the current batch
            for filename in batch_files:
                file_path = os.path.join(input_folder, filename)
                try:# Extract acquisition date from the filename
                    match = re.search(r"^.*?_.*?_.*?_(\d{8})_", filename)
                    if match:
                        date_str = match.group(1)
                        date = datetime.strptime(date_str, "%Y%m%d")
                    else:
                        skipped_files += 1
                        print(f"Warning: Could not extract date from {filename}. Skipping.")
                        pbar.update(1)  # Increment progress for skipped file
                        continue
                    #print(f"Skipped {skipped_files} files in batch {i // batch_size + 1}.")
                    # Load raster file
                    #try:
                    raster = rioxarray.open_rasterio(file_path, masked=True, chunks={"x": 1000, "y": 1000})
                    # Reproject raster to EPSG:4326
                    if raster.rio.crs != "EPSG:4326":
                        #print(f"Reprojecting raster: {filename} to EPSG:4326")
                        raster = raster.rio.reproject("EPSG:4326")
                    
                    raster = raster.squeeze().rename({"x": "x", "y": "y"})
                    #raster_padded = raster.combine_first(reference_raster)
                    raster_padded = raster.reindex_like(reference_raster, method="nearest", fill_value=np.nan)
                    # Align raster to reference grid
                    #raster = raster.reindex_like(reference_raster, method="nearest", fill_value=np.nan)
                    batch_data.append(raster_padded)
                    
                    #date_str = re.search(r"_(\d{8})_", filename).group(1)
                    batch_time_index.append(date)
                    #except Exception as e:
                        #print(f"Error loading {filename}: {e}")
                    pbar.update(1)
                        #continue

                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    pbar.update(1)
                    continue
            # Combine batch into a dataset
            if batch_data:
                #print(f"Batch {i // batch_size + 1} has {len(batch_data)} rasters and {len(batch_time_index)} time indices.")
                if len(batch_time_index) != len(batch_data):
                    raise ValueError("Mismatch between time index and raster data in this batch.")
                
                batch_stack = xr.concat(batch_data, dim="time")
                #batch_stack["time"] = batch_time_index
                batch_stack["time"] = ("time", batch_time_index)  # Explicitly define the time dimension
                batch_stack["time"] = pd.to_datetime(batch_stack["time"])
                batch_stack.name = "LST"
                #print(batch_stack["time"])

                # Add geospatial metadata
                batch_stack.attrs.update({
                    "grid_mapping": "spatial_ref",  # Links CRS
                    "crs": "EPSG:4326",            # Set CRS as WGS84
                    "GeoTransform": f"{aoi_bbox['min_lon']} {resolution} 0 {aoi_bbox['max_lat']} 0 -{resolution}",
                })
                # Add dataset-level attributes (new or existing)
                if os.path.exists(output_file):
                    # For appending mode, retain existing dataset-level attributes
                    existing_ds = xr.open_dataset(output_file)
                    existing_attrs = existing_ds.attrs
                    existing_ds.close()
                else:
                    # If the file doesn't exist, define new dataset-level attributes
                    existing_attrs = {
                        "title": "Landsat Surface Temperature Dataset",
                        "institution": "Politecnico di Milano",
                        "source": "Landsat 8 Level-2 Surface Temperature Products",
                        "history": f"Created on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                        "references": "https://landsat.usgs.gov/",
                        "Conventions": "CF-1.7",
                        "comment": "This NetCDF file was generated using rioxarray and xarray.",
                    }

                batch_stack.attrs.update(existing_attrs)  # Ensure all attributes are present

                # # Add dataset-level attributes
                # batch_stack.attrs.update({
                #     "title": "Landsat Surface Temperature Dataset",
                #     "institution": "Politecnico di Milano",
                #     "source": "Landsat 8 Level-2 Surface Temperature Products",
                #     "history": f"Created on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                #     "references": "https://landsat.usgs.gov/",
                #     "Conventions": "CF-1.7",
                #     "comment": "This NetCDF file was generated using rioxarray and xarray."
                # })

                # Add CRS as a separate variable (required for QGIS compatibility)
                crs_var = xr.DataArray(
                    0,
                    name="spatial_ref",
                    attrs={
                        "grid_mapping_name": "latitude_longitude",
                        "crs_wkt": (
                            "GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\","
                            "SPHEROID[\"WGS 84\",6378137,298.257223563]],"
                            "PRIMEM[\"Greenwich\",0],UNIT[\"degree\",0.0174532925199433]]"
                        ),
                    },
                )

                # Set coordinates for NetCDF
                batch_stack.coords["spatial_ref"] = crs_var
                #batch_stack.coords["spatial_ref"] = 0  # Required for QGIS compatibility
                batch_stack["x"].attrs.update({"units": "degrees_east", "long_name": "Longitude"})
                batch_stack["y"].attrs.update({"units": "degrees_north", "long_name": "Latitude"})
                batch_stack["time"].attrs.update({"long_name": "Time", "standard_name": "time"})


                # Save batch to NetCDF
                #encoding = {"LST": {"zlib": True, "complevel": 5}}
                encoding = {"LST": {"zlib": False}}
                mode = 'w' if i == 0 else 'a'  # Write for first batch, append for others
                # Check dimensions before saving

                # if os.path.exists(output_file):
                #     existing_ds = xr.open_dataset(output_file)
                #     print(f"Existing NetCDF dimensions: {existing_ds.dims}")
                #     existing_ds.close()
                
                #print(f"Saving batch {i // batch_size + 1} to NetCDF...")

                start_time = time.time()
                batch_stack.to_netcdf(output_file, mode=mode, encoding=encoding, engine="netcdf4") #ADDED
                #batch_stack.to_netcdf(output_file, mode=mode, encoding=encoding, engine="netcdf4", unlimited_dims=["time"],)
                end_time = time.time()
                #print(f"Batch {i // batch_size + 1} saved in {end_time - start_time:.2f} seconds.")
                #print(f"Batch {i // batch_size + 1} processed successfully.")
            #print(f"Batch {i // batch_size + 1} processed. Skipped {skipped_files} files.")
    print(f"NetCDF file saved at: {output_file}")
    
def make_netcdf(inputFolder, outptNetCDF):
    import os
    import rioxarray
    import xarray as xr
    import functions

    # Initialize an empty list to store individual DataArrays
    data_arrays = []

    # Iterate over all TIFF files in the directory
    for tiff_file in sorted(os.listdir(inputFolder)):
        if tiff_file.endswith(".TIF"):
            file_path = os.path.join(inputFolder, tiff_file)
            
            # Extract the date from the filename
            date = functions.extract_date(tiff_file)
            print("Extracted date of current files:", date)
            # Read the TIFF file as an xarray DataArray
            da = rioxarray.open_rasterio(file_path)
            
            # Reproject to WGS84 (EPSG:4326)
            da = da.rio.reproject("EPSG:4326")
            
            # Drop the "band" dimension and rename to something meaningful
            da = da.squeeze("band", drop=True).rename("LST")
            
            # Add a time coordinate
            da = da.expand_dims(time=[date])
            
            # Add metadata
            da.attrs["long_name"] = "Land Surface Temperature"
            da.attrs["units"] = "Kelvin"

            # Append to the list
            data_arrays.append(da)

    # Combine all DataArrays along the "time" dimension
    combined = xr.concat(data_arrays, dim="time")

    # Ensure combined is a Dataset
    if isinstance(combined, xr.DataArray):
        combined = combined.to_dataset(name="LST")

    # Rename dimensions for clarity
    combined = combined.rename({"x": "lon", "y": "lat"})

    # Add metadata
    combined.attrs["title"] = "Land Surface Temperature"
    combined.attrs["description"] = "Dataset of combined TIFF images from Landsat 8 of the 10th band (LST) data reprojected to EPSG:4326."
    combined.attrs["crs"] = "EPSG:4326"
    combined.lon.attrs["units"] = "degrees_east"
    combined.lat.attrs["units"] = "degrees_north"

    # Add CF-compliant metadata to the LST variable
    if "LST" in combined.data_vars:
        combined["LST"].attrs["long_name"] = "Land Surface Temperature"
        combined["LST"].attrs["units"] = "Kelvin"
    else:
        print("Error: Variable 'LST' not found in the combined dataset.")
        raise KeyError("LST variable is missing.")

    # Ensure proper time encoding
    time_encoding = {"units": "days since 1970-01-01", "calendar": "gregorian"}

    # Save to NetCDF
    combined.to_netcdf(
        outptNetCDF,
        engine="netcdf4",
        encoding={"time": time_encoding}  # Apply encoding directly to 'time'
    )
    print(f"NetCDF file saved: {outptNetCDF}")

def make_netcdf3(inputFolder, outptNetCDF):
    import os
    import rioxarray
    import xarray as xr
    import functions

    # Initialize an empty list to store individual DataArrays
    data_arrays = []

    # Iterate over all TIFF files in the directory
    for tiff_file in sorted(os.listdir(inputFolder)):
        if tiff_file.endswith(".TIF"):
            file_path = os.path.join(inputFolder, tiff_file)
            
            # Extract the date from the filename
            date = functions.extract_date(tiff_file)
            print("Extracted date of current files:", date)
            # Read the TIFF file as an xarray DataArray
            da = rioxarray.open_rasterio(file_path)
            
            # Reproject to WGS84 (EPSG:4326)
            da = da.rio.reproject("EPSG:4326")
            
            # Drop the "band" dimension and rename to something meaningful
            da = da.squeeze("band", drop=True).rename("LST")
            
            # Add a time coordinate
            da = da.expand_dims(time=[date])
            
            # Verify the data structure
            #print(f"Processed DataArray for {tiff_file}:\n", da)

            # Add metadata
            da.attrs["long_name"] = "Land Surface Temperature"
            da.attrs["units"] = "Kelvin"

            # Append to the list
            data_arrays.append(da)

    # Combine all DataArrays along the "time" dimension
    combined = xr.concat(data_arrays, dim="time")
    # Ensure combined is a Dataset
    if isinstance(combined, xr.DataArray):
        combined = combined.to_dataset(name="LST")  # Convert to Dataset with variable name "LST"

    # Rename dimensions for clarity
    combined = combined.rename({"x": "lon", "y": "lat"})

    # Debugging: Inspect the combined dataset
    print("Combined dataset structure before conversion:", combined)
    print("Data variables in combined dataset:", combined.data_vars)

    # # Ensure `combined` is a Dataset
    # if isinstance(combined, xr.DataArray):
    #     # Convert to Dataset
    #     combined = combined.to_dataset(name="data")

    # Debugging: Check structure after ensuring Dataset
    #print("Combined dataset structure after renaming dimensions:", combined)
    

    # Add metadata
    combined.attrs["title"] = "Land Surface Temperature"
    combined.attrs["description"] = "Dataset of combined TIFF images from Landsat 8 of the 10th band (LST) data reprojected to EPSG:4326."
    combined.attrs["crs"] = "EPSG:4326"
    #combined.attrs["history"] = "Created on 2025-01-01"
   
    combined.lon.attrs["units"] = "degrees_east"
    combined.lat.attrs["units"] = "degrees_north"

   # Add CF-compliant metadata to the "LST" variable
    #combined["LST"].attrs["long_name"] = "Land Surface Temperature"
    #combined["LST"].attrs["units"] = "Kelvin"  # Update with the appropriate units

    # Add CF-compliant metadata to the LST variable
    if "LST" in combined.data_vars:
        combined["LST"].attrs["long_name"] = "Land Surface Temperature"
        combined["LST"].attrs["units"] = "Kelvin"  # Update with the appropriate units
    else:
        print("Error: Variable 'LST' not found in the combined dataset.")
        raise KeyError("LST variable is missing.")



    # Ensure proper time encoding
    time_encoding = {"time": {"units": "days since 1970-01-01", "calendar": "gregorian"}}
    # Ensure proper time metadata
    #combined["time"].attrs.update({"units": "days since 1970-01-01", "calendar": "gregorian"})
    #combined.to_netcdf(outptNetCDF, encoding={"time": time_encoding})
    combined.to_netcdf(outptNetCDF,engine="netcdf4",encoding={"time": time_encoding})  # Apply encoding directly to 'time')
    # Save to NetCDF
    combined.close()
    print(f"NetCDF file saved: {outptNetCDF}")

def make_netcdf_test(inputFolder, outptNetCDF):

    import os
    import rioxarray
    import xarray as xr

    # Define the folder and output file
    input_folder = inputFolder
    output_file = outptNetCDF

    # List all TIFF files in the folder
    tiff_files = [f for f in os.listdir(input_folder) if f.endswith(".TIF")]
    if not tiff_files:
        raise FileNotFoundError("No TIFF files found in the folder.")

    # Initialize an empty list to store reprojected DataArrays
    data_arrays = []

    for tiff_file in tiff_files:
        # Full path to the TIFF file
        file_path = os.path.join(input_folder, tiff_file)

        # Extract date from the filename (adjust indexing if needed)
        date_str = tiff_file.split('_')[3]
        date_formatted = f"{date_str[6:8]}/{date_str[4:6]}/{date_str[2:4]}"  # dd/mm/yy format

        # Open the TIFF file
        da = rioxarray.open_rasterio(file_path)

        # Reproject to EPSG:4326
        da = da.rio.reproject("EPSG:4326")

        # Remove the "band" dimension and rename it
        da = da.squeeze("band", drop=True).rename("LST")

        # Add a time dimension
        da = da.expand_dims(time=[date_formatted])

        # Add metadata
        da.attrs["long_name"] = "Land Surface Temperature"
        da.attrs["units"] = "Kelvin"

        # Append the DataArray to the list
        data_arrays.append(da)

    # Combine all DataArrays along the time dimension
    combined = xr.concat(data_arrays, dim="time")

    # Rename dimensions for clarity
    combined = combined.rename({"x": "lon", "y": "lat"})

    # Add global attributes to the dataset
    combined.attrs["title"] = "Land Surface Temperature Dataset"
    combined.attrs["description"] = "Landsat 8 LST data reprojected to EPSG:4326."
    combined.attrs["crs"] = "EPSG:4326"
    combined.attrs["source"] = "Landsat 8"

    # Save to a NetCDF file
    time_encoding = {"time": {"units": "days since 1970-01-01", "calendar": "gregorian"}}
    combined.to_netcdf(output_file, encoding={"time": time_encoding})

    print(f"NetCDF file saved: {output_file}")


