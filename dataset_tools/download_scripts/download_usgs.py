import json
import requests
import sys
import time
import datetime
import threading
import re
from loguru import logger
import argparse
import os

# Time range for the EO1H dataset
START_DATE = "2001-05-01"
END_DATE = "2017-03-12"

NLCD_DATA_PRODUCT_NAMES = dict(
    FracImp='Fractional Impervious Surface',
    LndCov='Land Cover',
    LndCovChg='Land Cover Change',
    SpecChgDOY='Spectral Change Day of Year ' # don't remove the space
)

DATASET_NAMES = {
    'nlcd': 'NLCD',
}

DATASET_PRODUCTS = {
    'nlcd': NLCD_DATA_PRODUCT_NAMES,
}

serviceUrl = "https://m2m.cr.usgs.gov/api/api/json/stable/"

max_results = 5000
maxthreads = 5 # Threads count for downloads
sema = threading.Semaphore(value=maxthreads)
label = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") # Customized label using date time
threads = []

try:
    with open('USGS_login.json', 'r') as f:
        login_info = json.load(f)
except:
    with open('dataset_tools/USGS_login.json', 'r') as f:
        login_info = json.load(f)

USERNAME = login_info['username']
TOKEN = login_info['token']

# send http request
def sendRequest(url, data, apiKey = None):  
    pos = url.rfind('/') + 1
    endpoint = url[pos:]
    json_data = json.dumps(data)
    
    if apiKey == None:
        response = requests.post(url, json_data)
    else:
        headers = {'X-Auth-Token': apiKey}              
        response = requests.post(url, json_data, headers = headers)    
    
    try:
      httpStatusCode = response.status_code 
      if response == None:
          logger.error("No output from service")
          sys.exit()
      output = json.loads(response.text)	
      if output['errorCode'] != None:
          logger.error(f"Failed Request ID {output['requestId']}")
          logger.error(f"{output['errorCode']} - {output['errorMessage']}")
          sys.exit()
      if  httpStatusCode == 404:
          logger.error("404 Not Found")
          sys.exit()
      elif httpStatusCode == 401: 
          logger.error("401 Unauthorized")
          sys.exit()
      elif httpStatusCode == 400:
          logger.error(f"Error Code {httpStatusCode}")
          sys.exit()
    except Exception as e: 
          response.close()
          pos=serviceUrl.find('api')
          logger.error(f"Failed to parse request {endpoint} response. Re-check the input {json_data}. The input examples can be found at {url[:pos]}api/docs/reference/#{endpoint}")
          sys.exit()
    response.close()    
    logger.info(f"Finished request {endpoint} with request ID {output['requestId']}")
    
    return output['data']

def downloadFile(url, path):
    sema.acquire()
    try:        
        response = requests.get(url, stream=True)
        disposition = response.headers['content-disposition']
        filename = re.findall("filename=(.+)", disposition)[0].strip("\"")
        logger.info(f"Downloading {filename} ...")
        if path != "" and path[-1] != "/":
            filename = "/" + filename
        open(path + filename, 'wb').write(response.content)
        logger.info(f"Downloaded {filename}")
        sema.release()
    except Exception as e:
        logger.error(f"Failed to download from {url}. {e}.")
        sema.release()
        # Don't retry automatically to avoid infinite loops
    
def runDownload(threads, url, path):
    thread = threading.Thread(target=downloadFile, args=(url, path))
    threads.append(thread)
    thread.start()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_date", type=str, default=START_DATE)
    parser.add_argument("--end_date", type=str, default=END_DATE)
    parser.add_argument("--dataset_name", type=str, required=True, choices=DATASET_NAMES.keys())
    parser.add_argument("--dataproduct_name", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="data/nlcd/")
    parser.add_argument("--max_results", type=int, default=5000)
    args = parser.parse_args()
    
    args.output_path = os.path.join(args.output_path, args.dataset_name)
    os.makedirs(args.output_path, exist_ok=True)

    # 1. Login to get the api key
    payload = {'username' : USERNAME, 'token' : TOKEN}
    apiKey = sendRequest(serviceUrl + "login-token", payload)

    datasetName = args.dataset_name
    payload = {'datasetName' : datasetName}

    logger.info("Searching datasets...")
    datasets = sendRequest(serviceUrl + "dataset-search", payload, apiKey)
    logger.info(f"Found {len(datasets)} datasets")
    
    # Filter the datasets
    if args.dataproduct_name is None:
        dataset_products_abbr = list(DATASET_PRODUCTS[args.dataset_name].keys())
    else:
        assert args.dataproduct_name in DATASET_PRODUCTS[args.dataset_name].keys(), f"Data product {args.dataproduct_name} not found in dataset {args.dataset_name}"
        dataset_products_abbr = [args.dataproduct_name]
    dataset_products_names = [DATASET_PRODUCTS[args.dataset_name][abbr] for abbr in dataset_products_abbr]
    datasets = [dataset for dataset in datasets if dataset['collectionName'] in dataset_products_names]
    logger.info(f"Keep {len(datasets)} datasets")

    # 2. Download the data
    acquisitionFilter = {'start' : args.start_date, 'end' : args.end_date}
    
    for dataset in datasets:
        sub_dataset_path = os.path.join(args.output_path, dataset['collectionName'].strip().replace(" ", "_").lower())
        os.makedirs(sub_dataset_path, exist_ok=True)
        
        payload = {'datasetName' : dataset['datasetAlias'],
            'maxResults' : args.max_results,
            'sceneFilter' : {
                'acquisitionFilter' : acquisitionFilter,
            }
        }

        # Now I need to run a scene search to find data to download
        logger.info("Searching scenes...")   
        scenes = sendRequest(serviceUrl + "scene-search", payload, apiKey)
        logger.info(f"Found {scenes['recordsReturned']} scenes")

        # 3. Download the data
        if scenes['recordsReturned'] > 0:
            # Aggregate a list of scene ids
            sceneIds = []
            for result in scenes['results']:
                # Add this scene to the list I would like to download
                sceneIds.append(result['entityId'])

            payload = {'datasetName' : dataset['datasetAlias'], 'entityIds' : sceneIds}                    
            downloadOptions = sendRequest(serviceUrl + "download-options", payload, apiKey)

            downloads = []
            total_size = 0
            for product in downloadOptions:
                # Make sure the product is available for this scene
                if product['available'] == True:
                    downloads.append({'entityId' : product['entityId'],
                                    'productId' : product['id']})
                    total_size += product['filesize']

            logger.info(f"Found {len(downloads)} scenes to download, total size: {total_size/1024**3:.2f} GB")

            if downloads:
                requestedDownloadsCount = len(downloads)
                # set a label for the download request
                label = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") # Customized label using date time
                payload = {'downloads' : downloads,
                                                'label' : label}
                # Call the download to get the direct download urls
                requestResults = sendRequest(serviceUrl + "download-request", payload, apiKey)          
                                
                # PreparingDownloads has a valid link that can be used but data may not be immediately available
                # Call the download-retrieve method to get download that is available for immediate download
                if requestResults['preparingDownloads'] != None and len(requestResults['preparingDownloads']) > 0:
                    payload = {'label' : label}
                    moreDownloadUrls = sendRequest(serviceUrl + "download-retrieve", payload, apiKey)
                    
                    downloadIds = []  
                    
                    for download in moreDownloadUrls['available']:
                        if str(download['downloadId']) in requestResults['newRecords'] or str(download['downloadId']) in requestResults['duplicateProducts']:
                            downloadIds.append(download['downloadId'])
                            runDownload(threads, download['url'], sub_dataset_path)
                        
                    for download in moreDownloadUrls['requested']:
                        if str(download['downloadId']) in requestResults['newRecords'] or str(download['downloadId']) in requestResults['duplicateProducts']:
                            downloadIds.append(download['downloadId'])
                            runDownload(threads, download['url'], sub_dataset_path)
                        
                    # Didn't get all of the requested downloads, call the download-retrieve method again probably after 30 seconds
                    previous_count = len(downloadIds)
                    max_wait_time = 3600  # Maximum wait time of 1 hour
                    start_time = time.time()
                    while len(downloadIds) < requestedDownloadsCount: 
                        # Check if we've exceeded the maximum wait time
                        if time.time() - start_time > max_wait_time:
                            logger.warning(f"Maximum wait time ({max_wait_time/60:.1f} minutes) exceeded. Stopping download loop.")
                            break
                            
                        preparingDownloads = requestedDownloadsCount - len(downloadIds)
                        logger.info(f"\n{preparingDownloads} downloads are not available. Waiting for 30 seconds.")
                        time.sleep(30)
                        logger.info("Trying to retrieve data")
                        moreDownloadUrls = sendRequest(serviceUrl + "download-retrieve", payload, apiKey)
                        for download in moreDownloadUrls['available']:                            
                            if download['downloadId'] not in downloadIds and (str(download['downloadId']) in requestResults['newRecords'] or str(download['downloadId']) in requestResults['duplicateProducts']):
                                downloadIds.append(download['downloadId'])
                                runDownload(threads, download['url'], sub_dataset_path)
                        
                        # If no new downloads were found in this iteration, break to avoid infinite loop
                        if len(downloadIds) == previous_count:
                            logger.warning("No new downloads found after waiting. Stopping download loop.")
                            break
                        previous_count = len(downloadIds)
                else:
                    # Get all available downloads
                    for download in requestResults['availableDownloads']:
                        runDownload(threads, download['url'], sub_dataset_path)
            else:
                logger.info("No available downloads found for this dataset.")
        else:
            logger.info("No scenes found for this dataset.")

    if threads:
        logger.info("Downloading files... Please do not close the program")
        for thread in threads:
            thread.join()
        logger.info("Complete Downloading")
    else:
        logger.info("No downloads were initiated. Exiting.")
                
    # Logout so the API Key cannot be used anymore
    endpoint = "logout"  
    if sendRequest(serviceUrl + endpoint, None, apiKey) == None:        
        logger.info("Logged Out")
    else:
        logger.info("Logout Failed")  