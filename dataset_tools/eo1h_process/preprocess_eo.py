import argparse
import concurrent.futures
import os
import shutil
import zipfile
from typing import Dict, List

from tqdm import tqdm

import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from eo1h_process.global_patching_utils import tile_image_with_grids

# Defined by Datt et al. 2003, Table IV
STABLE_BANDS: List[int] = (
    list(range(10, 58))
    + list(range(81, 98))
    + list(range(101, 120))
    + list(range(134, 165))
    + list(range(182, 222))
)

# Noisy bands to remove
UNSTABLE_BANDS: List[int] = (
    list(range(1, 10))
    + list(range(58, 81))
    + list(range(98, 101))
    + list(range(120, 134))
    + list(range(165, 182))
    + list(range(222, 243))
)


def ensure_directories(base_save_dir: str, region: str) -> Dict[str, str]:
    region_dir = os.path.join(base_save_dir, region)
    unzip_dir = os.path.join(region_dir, "released_data")
    patched_dir = os.path.join(region_dir, "patched_data")
    os.makedirs(unzip_dir, exist_ok=True)
    os.makedirs(patched_dir, exist_ok=True)
    return {
        "region_dir": region_dir,
        "unzip_dir": unzip_dir,
        "patched_dir": patched_dir,
    }


def list_zip_files(directory: str) -> List[str]:
    return sorted([f for f in os.listdir(directory) if f.lower().endswith(".zip")])


def safe_remove(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)


def process_zip_file(
    zip_filename: str,
    region_dir: str,
    unzip_dir: str,
) -> int:
    sub_dir = zip_filename.split(".")[0]
    sub_dir_path = os.path.join(unzip_dir, sub_dir)

    # Unzip lazily if not already extracted
    if not os.path.isdir(sub_dir_path) or not os.listdir(sub_dir_path):
        os.makedirs(sub_dir_path, exist_ok=True)
        with zipfile.ZipFile(os.path.join(region_dir, zip_filename), "r") as zip_ref:
            zip_ref.extractall(sub_dir_path)

    # Clean up: remove README and unstable bands if present
    safe_remove(os.path.join(sub_dir_path, "README.txt"))

    entity_id = sub_dir.split("_")[0]
    for band in UNSTABLE_BANDS:
        band_path = os.path.join(
            sub_dir_path, f"{entity_id}_B{band:03d}_L1T.TIF"
        )
        safe_remove(band_path)
    return sub_dir_path
        
def verify_patch_dir(patch_dirs: List[str]) -> bool:
    for patch_dir in patch_dirs:
        if len(os.listdir(patch_dir)) != len(STABLE_BANDS) + 1:
            return False
    return True

def unzip_patchify_verify(zip_filename: str, region_dir: str, unzip_dir: str, patched_dir: str, nodata_thresh: float):
    sub_dir_path = process_zip_file(zip_filename, region_dir, unzip_dir)
    stats, _ = tile_image_with_grids(sub_dir_path, patched_dir, nodata_rate_max=nodata_thresh)
    assert verify_patch_dir(stats["patch_paths"]), f"Patch directory {patched_dir} is incomplete."
    shutil.rmtree(sub_dir_path, ignore_errors=True)
    safe_remove(os.path.join(region_dir, zip_filename))
    return stats["kept"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/project/common/geospatial/EO1H",
        help="Base directory where region subfolder and zip files reside.",
    )
    parser.add_argument("--region", "-r", type=str, default="NP")
    parser.add_argument(
        "--workers",
        type=int,
        default=1, # Rasterio is not thread safe
        help="Number of worker threads to use for processing.",
    )
    parser.add_argument(
        "--nodata_thresh",
        type=float,
        default=0.1,
        help="No-data threshold for patch validity.",
    )
    args = parser.parse_args()

    dirs = ensure_directories(args.save_dir, args.region)
    region_dir = dirs["region_dir"]
    unzip_dir = dirs["unzip_dir"]
    patched_dir = dirs["patched_dir"]

    zip_files = list_zip_files(region_dir)
    if not zip_files:
        print(f"No .zip files found in {region_dir}. Nothing to process.")
        return

    total_patches = 0
    with tqdm(total=len(zip_files), desc="Processing files") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            future_to_zip = {
                ex.submit(
                    unzip_patchify_verify,
                    zf,
                    region_dir,
                    unzip_dir,
                    patched_dir,
                    args.nodata_thresh,
                ): zf
                for zf in zip_files
            }

            for future in concurrent.futures.as_completed(future_to_zip):
                zf_name = future_to_zip[future]
                try:
                    created = future.result()
                    total_patches += int(created)
                except Exception as exc:
                    # Keep going even if one file fails; report the failure
                    print(f"Failed processing {zf_name}: {exc}")
                finally:
                    pbar.update(1)
                    pbar.set_postfix({"patches": total_patches})

    # remove unzip dir
    shutil.rmtree(unzip_dir, ignore_errors=True)
    # mv all dirs in patched_dir to region_dir
    for dir in os.listdir(patched_dir):
        shutil.move(os.path.join(patched_dir, dir), region_dir)
    # remove patched_dir
    shutil.rmtree(patched_dir, ignore_errors=True)


if __name__ == "__main__":
    main()