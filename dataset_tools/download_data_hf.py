# from datasets import load_dataset

CONFIGS = ['AC-SWIR1', 'AC-SWIR2', 'AC-SWIR3', 'AC-SWIR4', 'AC-VNIR', 'AF-SWIR1', 'AF-SWIR2', 'AF-SWIR3', 'AF-SWIR4', 'AF-VNIR', 'EA-SWIR1', 'EA-SWIR2', 'EA-SWIR3', 'EA-SWIR4', 'EA-VNIR', 'EU-SWIR1', 'EU-SWIR2', 'EU-SWIR3', 'EU-SWIR4', 'EU-VNIR', 'LA-SWIR1', 'LA-SWIR2', 'LA-SWIR3', 'LA-SWIR4', 'LA-VNIR', 'NA-SWIR1', 'NA-SWIR2', 'NA-SWIR3', 'NA-SWIR4', 'NA-VNIR', 'OC-SWIR1', 'OC-SWIR2', 'OC-SWIR3', 'OC-SWIR4', 'OC-VNIR', 'SEA-SWIR1', 'SEA-SWIR2', 'SEA-SWIR3', 'SEA-SWIR4', 'SEA-VNIR', 'SWA-SWIR1', 'SWA-SWIR2', 'SWA-SWIR3', 'SWA-SWIR4', 'SWA-VNIR']


# def download_data():
#     for config_name in CONFIGS:
#         print(f"downloading {config_name}")
#         ds = load_dataset(
#             path="GFM-Bench/EO1H-313K",
#             name=config_name,
#             cache_dir="/home/jovyan/workspace/data/EO1H-313K"
#         )

from concurrent.futures import ProcessPoolExecutor, as_completed
from datasets import load_dataset, concatenate_datasets
from eo1h_to_hub import concat_regions, stack_groups_by_frame_id
# from download_data_hf import CONFIGS
from tqdm import tqdm, trange
from copy import deepcopy

import numpy as np

CHANNEL_GROUPS = ['SWIR1', 'SWIR2', 'SWIR3', 'SWIR4', 'VNIR']
REGIONS = ['AF', 'EA', 'EU', 'LA', 'NA', 'OC', 'SEA', 'SWA'] # no AC, AC veri done

def compare_dicts(dict1, dict2):
    for k, v in dict1.items():
        if v != dict2[k]:
            if np.isnan(v):
                assert np.isnan(dict2[k]) 
            else:
                assert False, f"got non-nan not equal"
        else:
            assert v == dict2[k]

def verify_each_region(region, position):
    dss = [
        load_dataset("GFM-Bench/EO1H-313K", name=f"{region}-{g}", cache_dir="~/workspace/data/EO1H-313K")
        for g in CHANNEL_GROUPS
    ]

    # verify total number of images
    length = len(dss[0]['train'])
    num_chan = 0
    for i in range(len(dss)):
        assert len(dss[i]['train']) == length
        num_chan += np.array(dss[i]['train'][0]['image']).shape[0]

    # verify total channels sum to 155
    assert num_chan == 155, f"expect 155 channels but got {num_chan}"

    # verify other features
    for i in trange(length, position=position, leave=True):
        base_timestamp_days = dss[0]['train'][i]['timestamp_days']
        base_frame_id = dss[0]['train'][i]['frame_id']
        base_geo = dss[0]['train'][i]['geo']

        for j in range(len(dss)):
            assert dss[j]['train'][i]['timestamp_days'] == base_timestamp_days, f"at {i}-th data, expect {base_timestamp_days} but got {dss[j]['train'][i]['timestamp_days']} in {j}-th ds"
            assert dss[j]['train'][i]['frame_id'] == base_frame_id,  f"at {i}-th data, expect {base_frame_id} but got {dss[j]['train'][i]['frame_id']} in {j}-th ds"
            # assert dss[j]['train'][i]['geo'] == base_geo,  f"at {i}-th data, expect {base_geo} but got {dss[j]['train'][i]['geo']} in {j}-th ds"
            compare_dicts(base_geo, dss[j]['train'][i]['geo'])
        
    print(f"{region} passed check")

def run_parallel_verification(regions):
    results = {}
    positions = range(len(regions))
    # Use as many processes as CPU cores by default
    with ProcessPoolExecutor() as executor:
        # Submit all regions to the executor
        futures = {executor.submit(verify_each_region, region, position): region for region, position in zip(regions, positions)}
        
        # Use tqdm to track progress
        for future in tqdm(as_completed(futures), total=len(futures), desc="Verifying regions"):
            region = futures[future]
            try:
                future.result()  # will raise exception if verify_each_region fails
                results[region] = "PASS"
            except Exception as e:
                results[region] = f"FAIL: {e}"
    
    return results

if __name__ == "__main__":
    results = run_parallel_verification(REGIONS)
    for region, status in results.items():
        print(f"{region}: {status}")