# Functions.py require a list of IDs
import json
import requests
import pandas as pd
import geopandas as gpd
import threading
import os
import sys
import re
import time
import datetime
from getpass import getpass

def downloadMain(url, maxThreads, downloadFileType, search_payload, datasetName, serviceUrl, bandNames, apiKey, fileGroupIds, data_dir): #data_dir == downloaDirectory
    
    sema = threading.Semaphore(value=maxThreads)
    label = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") # Customized label using date time
    threads = []
    
    #fileType = "band"  # line 13 tutorial
    scenes = sendRequest(serviceUrl + "scene-search", search_payload, apiKey)
    pd.json_normalize(scenes['results']) #CHECK
    # EXTRACT UNIQUE IDENTIFIERS FROM ALL SCENES
    entityIds = search_payload.get("entityIds", [])
    # Add scenes to the list
    listId = search_payload["listId"]
    scn_list_add_payload = {
        "listId": listId,
        "idField": "entityId",
        "entityIds": entityIds,
        "datasetName": datasetName,
    }

    idField = search_payload["idField"]
    print(idField)
    # Send the request to add scenes to the list
    if idField == "displayId":
        scn_list_add_payload["idField"] = idField
        print("Using 'displayId' for scene-list-add payload.")
    print(scn_list_add_payload)
    
    count = sendRequest(serviceUrl + "scene-list-add", scn_list_add_payload, apiKey)
    print(f"Added {count} scenes to list {listId}")
    
    # nonBulkCount = 0 #added as a test
    # for result in scenes["results"]:
    #     # Add this scene to the list I would like to download if bulk is available
    #     if result["options"]["bulk"] == True:
    #         entityIds.append(result[idField])
    #     else:
    #         nonBulkCount += 1
    #         #potrebbe essere utile printare l'id dei dati che non hanno bulk disponibile
    # print("Entity Ids in downloadMain", entityIds)
    # print("nonBulkCount in downloadMain", nonBulkCount)

    # listId = f"temp_{datasetName}_list"  # customized list id
    # scn_list_add_payload = {
    #     "listId": listId,
    #     "idField": idField,
    #     "entityIds": entityIds,
    #     "datasetName": datasetName,
    # }
    # scn_list_add_payload
    # count = sendRequest(serviceUrl + "scene-list-add", scn_list_add_payload, apiKey)
    # print("Count in downloadMain", count)
    # #sendRequest(serviceUrl + "scene-list-get", {'listId' : scn_list_add_payload['listId']}, apiKey) 

    download_opt_payload = {"listId": listId, "datasetName": datasetName}
    if downloadFileType == 'band_group':
        download_opt_payload['includeSecondaryFileGroups'] = True

    print(download_opt_payload)

    products = sendRequest(serviceUrl + "download-options", download_opt_payload, apiKey)
    print("Download options retrieved.")
    # pd.json_normalize(products)

    # filegroups = sendRequest(serviceUrl + "dataset-file-groups", {'datasetName' : datasetName}, apiKey)  
    # pd.json_normalize(filegroups['secondary'])

    download_request_results, secondaryListId  = SelectProductsForDownloading(downloadFileType, products, datasetName, serviceUrl, bandNames, apiKey, fileGroupIds, label)
    downloadRetrive(download_request_results, label, serviceUrl, apiKey, threads, data_dir, sema)

    remove_scnlst_payload = {
        "listId": listId
    }
    sendRequest(serviceUrl + "scene-list-remove", remove_scnlst_payload, apiKey)

    if downloadFileType == "band_group" and secondaryListId:
        remove_scnlst2_payload = {"listId": secondaryListId}
        sendRequest(serviceUrl + "scene-list-remove", remove_scnlst2_payload, apiKey)
    
    #maxThreads = 5  # Threads count for downloads
    #sema = threading.Semaphore(value=maxThreads)
    #label = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    

# STARTS HERE-------------------------------------------------------
def downloadFile(url, data_dir, sema, threads):
    sema.acquire()
    try:
        response = requests.get(url, stream=True)
        disposition = response.headers["content-disposition"]
        filename = re.findall("filename=(.+)", disposition)[0].strip('"')
        print(f"    Downloading: {filename}...")

        open(os.path.join(data_dir, filename), "wb").write(response.content)
        sema.release()
    except Exception as e:
        print(f"\nFailed to download from {url}. Will try to re-download.")
        sema.release()
        runDownload(threads, url, data_dir, sema)


def runDownload(threads, url, data_dir, sema):
    thread = threading.Thread(target=downloadFile, args=(url, data_dir, sema, threads))
    threads.append(thread)
    thread.start()


def setupOutputDir(downloadDirectory):
    data_dir = os.path.join(downloadDirectory, 'data')
    utils_dir = os.path.join(downloadDirectory, 'utils')
    dirs = [data_dir, utils_dir]

    for d in dirs:
        if not os.path.exists(d):
            try:
                os.makedirs(d)
                print(f"Directory '{d}' created successfully.")
            except OSError as e:
                print(f"Error creating directory '{d}': {e}")
        else:
            print(f"Directory '{d}' already exists.")


def sendRequest(url, search_payload, apiKey, exitIfNoResponse=True):
    """
    Send a request to an M2M endpoint and returns the parsed JSON response.

    Parameters:
    endpoint_url (str): The URL of the M2M endpoint
    payload (dict): The payload to be sent with the request

    Returns:
    dict: Parsed JSON response
    """

    json_data = json.dumps(search_payload)

    if apiKey == None:
        response = requests.post(url, json_data)
    else:
        headers = {"X-Auth-Token": apiKey}
        response = requests.post(url, json_data, headers=headers)
    try:
        httpStatusCode = response.status_code
        if response == None:
            print("No output from service")
            if exitIfNoResponse:
                sys.exit()
            else:
                return False
        output = json.loads(response.text)
        if output["errorCode"] != None:
            print(output["errorCode"], "- ", output["errorMessage"])
            if exitIfNoResponse:
                sys.exit()
            else:
                return False
        if httpStatusCode == 404:
            print("404 Not Found")
            if exitIfNoResponse:
                sys.exit()
            else:
                return False
        elif httpStatusCode == 401:
            print("401 Unauthorized")
            if exitIfNoResponse:
                sys.exit()
            else:
                return False
        elif httpStatusCode == 400:
            print("Error Code", httpStatusCode)
            if exitIfNoResponse:
                sys.exit()
            else:
                return False
    except Exception as e:
        response.close()
        print(e)
        if exitIfNoResponse:
            sys.exit()
        else:
            return False
    response.close()

    return output["data"]

def SelectProductsForDownloading(downloadFileType, products, datasetName, serviceUrl, bandNames, apiKey, fileGroupIds, label):
        # Select products
    print("Selecting products...")
    downloads = []
    secondaryListId = None
    if downloadFileType == 'bundle':
        # Select bundle files
        print("    Selecting bundle files...")
        for product in products:        
            if product["bulkAvailable"] and product['downloadSystem'] != 'folder':               
                downloads.append({"entityId":product["entityId"], "productId":product["id"]})


    elif downloadFileType == 'band':
        # Select band files
        print("    Selecting band files...")
        for product in products:  
            if product["secondaryDownloads"] is not None and len(product["secondaryDownloads"]) > 0:
                for secondaryDownload in product["secondaryDownloads"]:
                    for bandName in bandNames:
                        if secondaryDownload["bulkAvailable"] and bandName in secondaryDownload['displayId']:
                            downloads.append({"entityId":secondaryDownload["entityId"], "productId":secondaryDownload["id"]})


    elif downloadFileType == 'band_group':        
        # Get secondary dataset ID and file group IDs with the scenes
        print("    Checking for scene band groups and get secondary dataset ID and file group IDs with the scenes...")
        sceneFileGroups = []
        entityIds = []
        datasetId = None
        for product in products:  
            if product["secondaryDownloads"] is not None and len(product["secondaryDownloads"]) > 0:
                for secondaryDownload in product["secondaryDownloads"]:
                    if secondaryDownload["bulkAvailable"] and secondaryDownload["fileGroups"] is not None:
                        if datasetId == None:
                            datasetId = secondaryDownload['datasetId']
                        for fg in secondaryDownload["fileGroups"]:                            
                            if fg not in sceneFileGroups:
                                sceneFileGroups.append(fg)
                            if secondaryDownload['entityId'] not in entityIds:
                                entityIds.append(secondaryDownload['entityId'])

        # Send dataset request to get the secondary dataset name by the dataset ID
        data_req_payload = {
            "datasetId": datasetId,
        }
        results = sendRequest(serviceUrl + "dataset", data_req_payload, apiKey)
        secondaryDatasetName = results['datasetAlias']

        # Add secondary scenes to a list
        secondaryListId = f"temp_{datasetName}_scecondary_list" # customized list id
        sec_scn_add_payload = {
            "listId": secondaryListId,
            "entityIds": entityIds,
            "datasetName": secondaryDatasetName
        }
        
        print("    Adding secondary scenes to list...")
        count = sendRequest(serviceUrl + "scene-list-add", sec_scn_add_payload, apiKey) 
        print("    Added", count, "secondary scenes\n")

        # Compare the provided file groups Ids with the scenes' file groups IDs
        if fileGroupIds:
            fileGroups = []
            for fg in fileGroupIds:
                fg = fg.strip() 
                if fg in sceneFileGroups:
                    fileGroups.append(fg)
        else:
            fileGroups = sceneFileGroups
    else:
        # Select all available files
        for product in products:        
            if product["bulkAvailable"]:
                if product['downloadSystem'] != 'folder':            
                    downloads.append({"entityId":product["entityId"], "productId":product["id"]})
                if product["secondaryDownloads"] is not None and len(product["secondaryDownloads"]) > 0:
                    for secondaryDownload in product["secondaryDownloads"]:
                        if secondaryDownload["bulkAvailable"]:
                            downloads.append({"entityId":secondaryDownload["entityId"], "productId":secondaryDownload["id"]})

    if downloadFileType != 'band_group':
        download_req2_payload = {
            "downloads": downloads,
            "label": label
        }
    else:
        if len(fileGroups) > 0:
            download_req2_payload = {
                "dataGroups": [
                    {
                        "fileGroups": fileGroups,
                        "datasetName": secondaryDatasetName,
                        "listId": secondaryListId
                    }
                ],
                "label": label
            }
        else:
            print('No file groups found')
            sys.exit()

    print(f"Sending download request ...")
    download_request_results = sendRequest(serviceUrl + "download-request", download_req2_payload, apiKey)
    print(f"Done sending download request") 
    
    if len(download_request_results['newRecords']) == 0 and len(download_request_results['duplicateProducts']) == 0:
        print('No records returned, please update your scenes or scene-search filter')
        sys.exit()

    return download_request_results, secondaryListId # CHECK

def downloadRetrive(download_request_results, label, serviceUrl, apiKey, threads, data_dir, sema):
    # Attempt the download URLs
    for result in download_request_results['availableDownloads']:
        #print(f"Get download url: {result['url']}\n" )
        runDownload(threads, result['url'], data_dir, sema)
        
    preparingDownloadCount = len(download_request_results['preparingDownloads'])
    preparingDownloadIds = []
    if preparingDownloadCount > 0:
        for result in download_request_results['preparingDownloads']:  
            preparingDownloadIds.append(result['downloadId'])

        download_ret_payload = {"label" : label}                
        # Retrieve download URLs
        print("Retrieving download urls...\n")
        download_retrieve_results = sendRequest(serviceUrl + "download-retrieve", download_ret_payload, apiKey, False)
        if download_retrieve_results != False:
            print(f"    Retrieved: \n" )
            for result in download_retrieve_results['available']:
                if result['downloadId'] in preparingDownloadIds:
                    preparingDownloadIds.remove(result['downloadId'])
                    runDownload(threads, result['url'], data_dir, sema)
                    print(f"       {result['url']}\n" )
                
            for result in download_retrieve_results['requested']:   
                if result['downloadId'] in preparingDownloadIds:
                    preparingDownloadIds.remove(result['downloadId'])
                    runDownload(threads, result['url'], data_dir, sema)
                    print(f"       {result['url']}\n" )
        
        # Didn't get all download URLs, retrieve again after 30 seconds
        while len(preparingDownloadIds) > 0: 
            print(f"{len(preparingDownloadIds)} downloads are not available yet. Waiting for 30s to retrieve again\n")
            time.sleep(30)
            download_retrieve_results = sendRequest(serviceUrl + "download-retrieve", download_ret_payload, apiKey, False)
            if download_retrieve_results != False:
                for result in download_retrieve_results['available']:                            
                    if result['downloadId'] in preparingDownloadIds:
                        preparingDownloadIds.remove(result['downloadId'])
                        #print(f"    Get download url: {result['url']}\n" )
                        runDownload(threads, result['url'], data_dir, sema)
                        
    print("\nDownloading files... Please do not close the program\n")
    for thread in threads:
        thread.join()

