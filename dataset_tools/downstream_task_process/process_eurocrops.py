import os
import sys
import glob
import tqdm
from multiprocessing import Pool, cpu_count

# find all shp files in root
root = "../data/EuroCrop"
save_dir = "../data/EuroCrop/EuroCrop_raw"
shp_files = glob.glob(os.path.join(root, "**/*.shp"), recursive=True)
paths = []
# for all the shp files, find their parent parent dir
for shp_file in shp_files:
    file_name = os.path.basename(shp_file).replace(".shp", ".tif")
    save_path = os.path.join(save_dir, file_name)
    paths.append((shp_file, save_path))
# remove duplicates
paths = list(set(paths))
# print(paths)

def process_shapefile(args):
    """Worker function to process a single shapefile"""
    shp_file, save_path = args
    cmd = f"python shp2tif.py {shp_file} {save_path} --attribute EC_hcat_c --pixel-size 30 --all-touched"
    return os.system(cmd)

# run the shp2tif.py script for each path in parallel
if __name__ == "__main__":
    num_processes = 4
    print(f"Processing {len(paths)} shapefiles using {num_processes} processes...")
    
    with Pool(processes=num_processes) as pool:
        # Use imap for progress tracking
        results = list(tqdm.tqdm(
            pool.imap(process_shapefile, paths),
            total=len(paths),
            desc="Processing shapefiles"
        ))
    
    # Check for any errors
    failed_count = sum(1 for result in results if result != 0)
    if failed_count > 0:
        print(f"Warning: {failed_count} files failed to process")
    else:
        print("All files processed successfully!")
