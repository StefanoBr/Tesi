# Import Packages
import requests
import sys
import geopandas as gpd
import xarray as rxr
import glob
import functionsForDownload
import functionsForPreprocessing
import functions
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import os

#LOGIN SECTION ----------------------------------------------------------------------------------------------------------------------------------
# Define File Paths and Token
token_file_path = r"C:\Users\Stefano\Desktop\3_Semester\TESI\USGS\USGS_TOKEN.txt"
with open(token_file_path, "r") as file:
    token = file.read().strip()

# Define EE M2M API Endpoint
serviceUrl = "https://m2m.cr.usgs.gov/api/api/json/stable/"
login_url = serviceUrl + "login-token"

# Authenticate
payload = {"username": "Stefano98", "token": token}
response = requests.post(login_url, json=payload)
login_results = response.json()

if login_results.get("errorCode"):
    print("Login Error:", login_results["errorMessage"])
    sys.exit(1)
else:
    apiKey = login_results["data"]
    print("Login successful, apiKey:", apiKey)

# PARAMETERS DEFINITION SECTION ----------------------------------------------------------------------------------------------------------------------------------
# Define Parameters
shapefile_path = r"C:\Users\Stefano\Desktop\3_Semester\TESI\BBOX\BB_MetropolitanCityOfMilan.shp"
output_dir = r"E:\TESI\USGS\DataFilteredByCC"
lowerLeft, upperRight, aoi = functions.computeBBOX(shapefile_path)
datasetName = "landsat_ot_c2_l2"
catalogNode = "EE"
startDate = "2015-06-27"
endDate = "2024-10-31"
filterid = "61af9273566bb9a8"
downloadDirectory = r"E:\TESI\USGS\QA_PIXEL2"
downloadFileType = "band"
bandNames = {"QA_PIXEL"}
fileGroupIds = {"ls_c2l2_st_band"}  # Surface temperature
maxThreads = 5
areaThreshold = 90

# DOWNLOAD SECTION ----------------------------------------------------------------------------------------------------------------------------------

# Filter Setup
dataset_filters_payload = {"datasetName": datasetName, "catalog": catalogNode}
filters_url = serviceUrl + "dataset-filters"
filters_response = requests.post(filters_url, headers={"X-Auth-Token": apiKey}, json=dataset_filters_payload)
filters_results = filters_response.json()
filterId = filters_results["data"][5]["id"]

# Pagination Logic for Scene Retrieval
search_url = serviceUrl + "scene-search"
headers = {"X-Auth-Token": apiKey}
starting_number = 0
page_size = 100
all_scenes = []

while True:
    # Build search payload with filters
    search_payload = {
        "datasetName": datasetName,
        "catalog": catalogNode,
        "sceneFilter": {
            "spatialFilter": {
                "filterType": "mbr",
                "lowerLeft": lowerLeft,
                "upperRight": upperRight,
            },
            "acquisitionFilter": {"start": startDate, "end": endDate},
            "metadataFilter": {"filterType": "value", "filterId": filterId, "value": "8"},
        },
        "startingNumber": starting_number,
        "maxResults": page_size,
    }

    # Send API request
    response = requests.post(search_url, headers=headers, json=search_payload)
    search_results = response.json()

    # Handle errors
    if search_results.get("errorCode"):
        print(f"Search Error: {search_results['errorMessage']}")
        break

    # Collect results
    scenes = search_results.get("data", {}).get("results", [])
    all_scenes.extend(scenes)
    print(f"Retrieved {len(scenes)} scenes (starting from {starting_number}).")

    if len(scenes) < page_size:  # Break if fewer results than page size
        break
    starting_number += page_size

print(f"Total scenes retrieved: {len(all_scenes)}")

# Setup Download Directory
functionsForDownload.setupOutputDir(downloadDirectory)
# Split scene_ids into batches of 100
scene_ids = [scene["entityId"] for scene in all_scenes]  # Collect scene IDs
batch_size = 100
scene_batches = [scene_ids[i:i + batch_size] for i in range(0, len(scene_ids), batch_size)] #same as scene_ids bt divided in [] brachets

print(f"Total scenes to download: {len(scene_ids)}")
print(f"Total batches to process: {len(scene_batches)}")
#print("Scene IDs", scene_ids)
#print("Batches", scene_batches)

# Iterate over batches and download
for batch_num, batch in enumerate(scene_batches):
    #print(f"Processing batch {batch_num + 1}/{len(scene_batches)} with {len(batch)} scenes.")
    
    # Create a payload for the current batch
    batch_payload = {
        "datasetName": datasetName,
        "catalog": catalogNode,
        "listId": f"temp_landsat_list_{batch_num}",
        "entityIds": batch,
        # "sceneFilter": {
        #     "metadataFilter": {"filterType": "value", "filterId": filterId, "value": "8"},
        # },
    }
    print("batch_payload", batch_payload)
    # Call the download function for the current batch
    functionsForDownload.downloadMain(
        login_url,
        maxThreads,
        downloadFileType,
        batch_payload,
        datasetName,
        serviceUrl,
        bandNames,
        apiKey,
        fileGroupIds,
        downloadDirectory,
    )

# PREPROCESSING SECTION ----------------------------------------------------------------------------------------------------------------------------------
# CLIPPING AND SAVING  -----------------------------------------------------------------------------------------------------------------------------------
# # Create output directory if not exists
# os.makedirs(output_dir, exist_ok=True)
# # Get list of raster files that ends with _QA_PIXEL.TIF (in case you have other files in there)
# raster_files = glob.glob(os.path.join(downloadDirectory, "*_QA_PIXEL.TIF"))
# # Get list of raster files
# raster_files = [
#     file for file in glob.glob(os.path.join(downloadDirectory, "*_QA_PIXEL.TIF"))
#     if os.path.isfile(file)  # Ensure only files are included
# ]
# # print("Processing the following files:")
# # for file in raster_files:
# #     print(file)
# print(f"Number of raster files to process: {len(raster_files)}")
# # Process rasters in parallel
# process_function = partial(functionsForPreprocessing.process_raster, aoi, output_dir)
# with ThreadPoolExecutor(max_workers=8) as executor:
#     results = list(executor.map(process_function, raster_files))
# for raster_name, cloud_cov in results:
#     if cloud_cov is not None:
#         print(f"{raster_name}: Cloud Coverage = {cloud_cov:.2f}%")
#     else:
#         print(f"{raster_name}: Processing failed.")

# COMPUTING OVERLP %  -----------------------------------------------------------------------------------------------------------------------------------
imagesAboveThreshold, files_above_threshold = functionsForPreprocessing.compute_overlapping_percentage_with_mask(output_dir, aoi, areaThreshold)
#print(files_above_threshold)
functions.plot_temporal_gaps(files_above_threshold)

#LOGOUT SECTION ----------------------------------------------------------------------------------------------------------------------------------
# Logout
logout_url = serviceUrl + "logout"
if functionsForDownload.sendRequest(logout_url, None, apiKey) is None:
    print("\nLogged Out\n")
else:
    print("\nLogout Failed\n")