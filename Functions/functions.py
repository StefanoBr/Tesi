# Functions.py
import json
import requests
import geopandas as gpd
#import threading
import os
#import sys
import rioxarray




# Landsat 8-9 OLI/TIRS C2 L2


def list_datasets(token, service_url, catalog_node="EE"):
    """
    List available datasets in a given catalog node.

    Parameters:
        token (str): The API token for authentication.
        service_url (str): The base URL of the USGS API.
        catalog_node (str): Catalog node, default is 'EE' (Earth Explorer).

    Returns:
        list: A list of available dataset names.
    """
    # Define dataset listing URL
    dataset_url = service_url + "dataset-search"
    dataset_payload = {"datasetName": None, "catalog": catalog_node, "publicOnly": True}

    # Send request to list available datasets
    headers = {"X-Auth-Token": token}
    response = requests.post(dataset_url, headers=headers, json=dataset_payload)

    # print("Status code:", response.status_code)  # Print status code for debugging
    # print("Response text:", response.text)  # Print full response for debugging

    dataset_results = response.json()
    # print("Dataset API response:", json.dumps(dataset_results, indent=2))

    # Check for errors in the response
    if dataset_results.get("errorCode"):
        print("Dataset Listing Error:", dataset_results["errorMessage"])
        return []
    # Process 'data' field if it contains datasets
    data = dataset_results.get("data")
    if data and isinstance(data, list):  # Ensure data is a list
        datasets = [
            dataset.get("collectionName")
            for dataset in data
            if "collectionName" in dataset
        ]
        print("Available datasets:", datasets)
        return datasets
    else:
        print("No datasets found or unexpected data structure.")
        return []

def computeBBOX(shapefile_path):

    gdf = gpd.read_file(shapefile_path)
    # print("Pre trnsformation", aoi_geodf.crs)
    gdf = gdf.to_crs("EPSG:4326")
    # print("Post trnsformation", gdf.crs)
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]

    # Define lower-left and upper-right points from the bounding box
    ll = {"longitude": bounds[0], "latitude": bounds[1]}
    ur = {"longitude": bounds[2], "latitude": bounds[3]}
    return ll, ur, gdf

def list_filters(datasetName, catalogNode, serviceUrl, token):
    # Define the payload for dataset-filters request
    dataset_filters_payload = {
        "datasetName": datasetName,  # Ensure this is your correct dataset name
        "catalog": catalogNode,  # Earth Explorer catalog
    }

    # URL for the dataset-filters endpoint
    filters_url = serviceUrl + "dataset-filters"

    # Send request to get dataset filters
    filters_response = requests.post(
        filters_url, headers={"X-Auth-Token": token}, json=dataset_filters_payload
    )
    filters_results = filters_response.json()

    # Check for errors
    if filters_results.get("errorCode"):
        print("Filter Retrieval Error:", filters_results["errorMessage"])
    else:
        # Print available filters for inspection
        print("Dataset Filters:", json.dumps(filters_results, indent=2))
    print(filters_results.get("data"))

def filterToSelectLandsat(datasetName, catalogNode, serviceUrl, token):
    dataset_filters_payload = {
        "datasetName": datasetName,
        "catalog": catalogNode,
    }
    filters_url = serviceUrl + "dataset-filters"
    filters_response = requests.post(
        filters_url, headers={"X-Auth-Token": token}, json=dataset_filters_payload
    )
    filters_results = filters_response.json()
    filterId = filters_results["data"][5]["id"]
    # print(f"Selected Satellite Filter ID: {filterId}")
    return filterId

def retriveBandsName(search_results, token, serviceUrl, datasetName):
    scenes = search_results.get("data", {}).get("results", [])
    first_scene = scenes[0]
    scene_id = first_scene.get("entityId")
    metadata_url = serviceUrl + "scene-metadata"
    metadata_payload = {
        "datasetName": datasetName,
        "entityIds": [scene_id],  # Requesting metadata for this specific scene
    }

    print("\nProcessing First Scene ID:", first_scene.get("entityId"))
    metadata = first_scene.get("metadata", [])
    headers = {"X-Auth-Token": token}
    # Send the metadata request
    metadata_response = requests.post(
        metadata_url, headers=headers, json=metadata_payload
    )
    metadata_results = metadata_response.json()

    # Check if metadata contains detailed band information
    if metadata_results.get("errorCode"):
        print("Metadata Retrieval Error:", metadata_results["errorMessage"])
    else:
        # Inspect metadata for band information
        metadata_data = (
            metadata_results.get("data", {}).get(scene_id, {}).get("bands", [])
        )
        if metadata_data:
            print("Band Information for Scene ID", scene_id)
            for band in metadata_data:
                print(
                    f" - Band Name: {band.get('name')}, Description: {band.get('description')}"
                )
        else:
            print("No band information found in detailed metadata.")

    # if metadata:
    #     print("Bands Information for First Scene:")
    #     for item in metadata:
    #         if "Band" in item.get("fieldName", ""):
    #             print(f" - {item['fieldName']}: {item.get('value')}")
    # else:
    #     print("No detailed band information found in metadata for the first scene.")

def list_unique_crs(folder_path):
    import rasterio
    """
    List and print the unique CRSs of all .TIF files in a folder.
    
    Parameters:
    - folder_path: Path to the folder containing .TIF files.
    
    Prints:
    - A message listing the unique CRSs found.
    """
    crs_set = set()  # Use a set to store unique CRSs

    # Iterate over all files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".TIF"):
            file_path = os.path.join(folder_path, file_name)
            try:
                with rasterio.open(file_path) as src:
                    crs = src.crs  # Extract the CRS from the file
                    if crs is not None:
                        crs_set.add(crs.to_string())
            except Exception as e:
                print(f"Error reading {file_name}: {e}")

    # Print the unique CRSs
    if crs_set:
        print(f"You have {len(crs_set)} different CRS(s): {', '.join(crs_set)}")
    else:
        print("No valid CRS found in the folder.")

def print_data_info(file_path):
    import rasterio

    # Open the raster file using rasterio
    with rasterio.open(file_path) as src:
        # Get resolution
        resolution = src.res  # Resolution is returned as (x_res, y_res)
        crs = src.crs         # Coordinate Reference System
        bounds = src.bounds   # Spatial extent of the raster
        width, height = src.width, src.height  # Dimensions in pixels

    # Print the information
    print(f"Resolution: {resolution} meters per pixel")
    print(f"Coordinate Reference System: {crs}")
    print(f"Bounds: {bounds}")
    print(f"Raster dimensions: {width} pixels wide, {height} pixels tall")

def compare_max_coordinates(input_folder):
    """
    Extract the max x and y dimensions for all .TIF products in a folder and compare differences.

    Parameters:
    - input_folder: Path to the folder containing Landsat .TIF files.

    Returns:
    - None. Prints the dimensions of rasters with differences.
    """
    dimensions = []
    
    # Process all .TIF files
    for filename in sorted(os.listdir(input_folder)):
        if filename.endswith(".TIF"):
            file_path = os.path.join(input_folder, filename)
            
            try:
                raster = rioxarray.open_rasterio(file_path)
                dims = (raster.rio.width, raster.rio.height)  # Get x (width) and y (height)
                dimensions.append((filename, dims))
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue

    # Compare dimensions
    unique_dims = {}
    for filename, dims in dimensions:
        if dims not in unique_dims:
            unique_dims[dims] = []
        unique_dims[dims].append(filename)

    # Print results
    print("\n=== Unique Dimensions ===")
    for dims, files in unique_dims.items():
        print(f"Dimensions: x={dims[0]}, y={dims[1]} - {len(files)} files")
        print(f"Files: {', '.join(files[:5])} {'...' if len(files) > 5 else ''}")

def list_and_compare_QA_PIXEL_files(folder_path, entityIds):
    """
    List files ending with '_QA_PIXEL.TIF' in a folder, remove the '_QA_PIXEL.TIF' suffix,
    and compare with the provided entityIds list.

    Args:
        folder_path (str): Path to the folder containing the files.
        entityIds (list): List of entityIds to compare against.
    """
    # Step 1: List files ending with '_QA_PIXEL.TIF'
    qa_pixel_files = [
        f for f in os.listdir(folder_path) if f.endswith("_QA_PIXEL.TIF")
    ]

    # Step 2: Remove '_QA_PIXEL.TIF' from filenames
    cleaned_product_names = [f.replace("_QA_PIXEL.TIF", "") for f in qa_pixel_files]

    # Step 3: Compare with the provided entityIds list
    products_not_in_entityIds = [
        product for product in cleaned_product_names if product not in entityIds
    ]

    # Step 4: Print results
    print(f"Total QA_PIXEL files found in folder: {len(cleaned_product_names)}")
    print(f"Total entityIds in list: {len(entityIds)}")
    print(f"Products not in entityIds ({len(products_not_in_entityIds)}):")
    
    for product in products_not_in_entityIds:
        print(product)

def list_files_in_folder(folder_path, suffix):
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(suffix)]

def extract_date(filename):
        import re
        import pandas as pd
        match = re.search(r"_(\d{8})_", filename)
        if match:
            return pd.to_datetime(match.group(1), format="%Y%m%d")
        else:
            raise ValueError(f"Date not found in filename: {filename}")

# PLOTTING -------------------------------------------------------------------------------------------------------------------------------------------------------

def plotAoi(aoi_geodf):
    import folium

    centroid_lat = aoi_geodf.geometry.centroid.y.mean()
    centroid_lon = aoi_geodf.geometry.centroid.x.mean()
    m = folium.Map(
        location=[centroid_lat, centroid_lon],
        zoom_start=8,
        tiles="openstreetmap",
        width="90%",
        height="90%",
        attributionControl=False,
    )  # add n estimate of where the center of the polygon would be located\
    # for the location [latitude longitude]
    for _, r in aoi_geodf.iterrows():
        sim_geo = gpd.GeoSeries(r["geometry"]).simplify(tolerance=0.001)
        geo_j = sim_geo.to_json()
        geo_j = folium.GeoJson(
            data=geo_j,
            style_function=lambda x: {
                "fillColor": "blue",
                "color": "black",
                "weight": 2,
                "fillOpacity": 0.3,
            },
        )
        geo_j.add_to(m)
    m  # display map

def plot_temporal_gaps(files):
    import matplotlib.pyplot as plt
    from datetime import datetime
    import matplotlib.dates as mdates
    # Extract dates from file names
    # Extract dates from filenames
    dates = []
    for file in files:
        filename = file.split("\\")[-1]  # Extract filename from path
        date_str = filename.split("_")[3]  # Correctly extract the 3rd section (index 3)
        date = datetime.strptime(date_str, "%Y%m%d")  # Convert to datetime
        dates.append(date)
    
    # Sort dates
    dates.sort()

    # Plot data
    fig, ax = plt.subplots(figsize=(16, 4))  # Widen the plot and reduce height
    y_positions = [1] * len(dates)  # Y-axis positions for dots
    ax.plot(dates, y_positions, 'o', label="Data Points", markersize=5)

    # Check for gaps greater than 1 month
    for i in range(1, len(dates)):
        delta = (dates[i] - dates[i - 1]).days
        if delta > 60: # More than one month
            # Mark gaps with a red line
            ax.plot([dates[i - 1], dates[i]], [1, 1], color='red', linewidth=1)

    # Formatting the x-axis to display only specific dates
    ax.set_xticks(dates)  # Set x-axis ticks to only the dates in the list
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b-%Y"))  # Full date format
    plt.xticks(rotation=90, ha="right", fontsize=5)  # Rotate x-axis labels for readability

    # Reduce y-axis size
    ax.set_ylim(0.8, 1.2)  # Reduce the height of the y-axis

    # Add labels, legend, and grid
    ax.set_title("Temporal Gaps in Raster Files", fontsize=16)
    ax.set_xlabel("Date", fontsize=14)
    ax.set_yticks([])  # Remove y-axis values (not needed)
    ax.legend(fontsize=12, loc="upper left")
    ax.grid(axis='x', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

def plot_seasonal_temporal_gaps2(files):
    import matplotlib.pyplot as plt
    from datetime import datetime
    import matplotlib.dates as mdates

    # Extract dates from filenames
    dates = []
    for file in files:
        filename = file.split("\\")[-1]  # Extract filename from path
        date_str = filename.split("_")[3]  # Correctly extract the 3rd section (index 3)
        date = datetime.strptime(date_str, "%Y%m%d")  # Convert to datetime
        dates.append(date)
    
    # Sort dates
    dates.sort()

    # Categorize dates by season
    summer_dates = [date for date in dates if date.month in [6, 7, 8, 9]]  # June to September
    winter_dates = [date for date in dates if date.month in [12, 1, 2, 3]]  # December to March
    intermediate_dates = [date for date in dates if date.month in [4, 5, 10, 11]]  # Other months

    # Helper function to plot a single season
    def plot_single_season2(ax, dates, title):
        if not dates:
            ax.text(0.5, 0.5, "No Data", transform=ax.transAxes, ha='center', fontsize=14)
            return
        
        y_positions = [1] * len(dates)  # Y-axis positions for dots
        ax.plot(dates, y_positions, 'o', label="Data Points", markersize=5)
        #plt.xticks(rotation=90, ha="right", fontsize=5)  # Rotate x-axis labels for readability
        # Check for gaps greater than 60 days
        for i in range(1, len(dates)):
            delta = (dates[i] - dates[i - 1]).days
            if delta > 60:  # More than 60 days
                ax.plot([dates[i - 1], dates[i]], [1, 1], color='red', linewidth=1)

        # Formatting the x-axis to display only specific dates
        ax.set_xticks(dates)  # Set x-axis ticks to only the dates in the list
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b-%Y"))  # Full date format
        ax.tick_params(axis='x', rotation=90, labelsize=8)

        # Reduce y-axis size
        ax.set_ylim(0.8, 1.2)  # Reduce the height of the y-axis
        ax.set_yticks([])  # Remove y-axis values (not needed)

        # Add title, legend, and grid
        ax.set_title(title, fontsize=12)
        ax.grid(axis='x', linestyle='--', alpha=0.6)

    # Create subplots for the three seasons
    fig, axs = plt.subplots(3, 1, figsize=(12, 12), constrained_layout=True)

    # Plot for each season
    plot_single_season2(axs[0], summer_dates, "Summer (June to September)")
    plot_single_season2(axs[1], winter_dates, "Winter (December to March)")
    plot_single_season2(axs[2], intermediate_dates, "Intermediate Seasons (April, May, October, November)")

    # Add an x-axis label to the last plot
    axs[2].set_xlabel("Date", fontsize=14)

    # Show the plots
    plt.show()

def plot_seasonal_temporal_gaps(files):
    import matplotlib.pyplot as plt
    from datetime import datetime

    # Extract dates from filenames
    dates = []
    for file in files:
        filename = file.split("\\")[-1]  # Extract filename from path
        date_str = filename.split("_")[3]  # Correctly extract the 3rd section (index 3)
        date = datetime.strptime(date_str, "%Y%m%d")  # Convert to datetime
        dates.append(date)
    
    # Sort dates
    dates.sort()

    # Categorize dates by season
    summer_dates = [date for date in dates if date.month in [6, 7, 8, 9]]  # June to September
    winter_dates = [date for date in dates if date.month in [12, 1, 2, 3]]  # December to March
    intermediate_dates = [date for date in dates if date.month in [4, 5, 10, 11]]  # Other months

    # Helper function to plot a single season
    def plot_single_season(ax, dates, title, rearrange_winter=False, adjust_intermediate=False):
        if not dates:
            ax.text(0.5, 0.5, "No Data", transform=ax.transAxes, ha='center', fontsize=14)
            return
        
        # Prepare data for plotting: Extract years and adjusted x-axis positions
        years = [date.year for date in dates]
        if rearrange_winter:
            # Adjust for winter: December appears before January
            x_positions = [
                (0.9 * (date.month - 12) if date.month == 12 else 0.75 * date.month) + 1.5 * date.day / 100 #scale month and day part of the date to improve readability
                for date in dates
            ]
        elif adjust_intermediate:
            # Custom scaling for intermediate seasons: Reduce May-October distance by 3x
            x_positions = [
                ((date.month - 2) * 1 if date.month in [10, 11] else  # Shift October and November back
                (date.month + 2) * 1 if date.month in [4, 5] else  # Shift April and May forward
                date.month) + 1.5 * date.day / 100  # Keep other months as is
                for date in dates
            ]
        else:
            x_positions = [0.8 * date.month + 1.5 * date.day / 100 for date in dates] #scale month and day part of the date to improve readability

        # Scatter plot with years on y-axis and adjusted x-axis positions
        ax.scatter(x_positions, years, label="Data Points", color="blue", s=20)

        # Format x-axis with specific dates
        date_labels = [date.strftime("%d-%b") for date in dates]  # Format: "day-month"
        ax.set_xticks(x_positions)
        ax.set_xticklabels(date_labels, rotation=90, ha="right")
        ax.tick_params(axis='x', labelsize=7)  # Resize x-axis labels for readability

        # Add labels and grid
        ax.set_title(title, fontsize=12)
        ax.set_ylabel("Year", fontsize=10)
        ax.set_ylim(min(years) - 1, max(years) + 1)  # Shortened vertical spacing
        ax.grid(axis='both', linestyle='--', alpha=0.6)

    # Create subplots for the three seasons
    fig, axs = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)  # Reduced height

    # Plot for each season
    plot_single_season(axs[0], summer_dates, "Summer (June to September)")
    plot_single_season(axs[1], winter_dates, "Winter (December to March)", rearrange_winter=True)
    plot_single_season(axs[2], intermediate_dates, "Intermediate Seasons (April, May, October, November)", adjust_intermediate=True)

    # Add a common xlabel for the bottom plot
    axs[2].set_xlabel("Date (Day-Month)", fontsize=14)

    # Show the plots
    plt.show()
 #Once done, may i add  other  grids to the netcdf? How?


