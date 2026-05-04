import ee
import argparse
from loguru import logger

DATASET_NAMES = {
    'BNETD': '"BNETD/land_cover/v1"',
    'TreeMap2016': "USFS/GTAC/TreeMap/v2016",
    'TreeMap2020': "USFS/GTAC/TreeMap/v2020",
    'TreeMap2022': "USFS/GTAC/TreeMap/v2022",
    'ISDASOIL': "ISDASOIL/Africa/v1/texture_class",
    'GFC': "UMD/hansen/global_forest_change_2024_v1_12",
}

def main(args):
    ee.Authenticate()
    ee.Initialize(project="ee-sihaozhe31")
    try:
        collection = ee.ImageCollection(DATASET_NAMES[args.dataset_name])
        size = collection.size().getInfo()
        images = collection.toList(size)
    except:
        image = ee.Image(DATASET_NAMES[args.dataset_name])
        size = 1
        images = ee.List([image])
        
    for i in range(size):
        if isinstance(images, list):
            img = images[i]
        else:
            img = ee.Image(images.get(i))
        year = img.get('year').getInfo() or f"img_{i}"
        desc = f"{args.dataset_name}_{year}"
        if args.bands is not None:
            bands = args.bands
        else:
            bands = img.bandNames().getInfo()
        logger.info(f"Downloading {desc} with bands {bands}")
        img_sel = img.select(bands).toFloat()
        
        task = ee.batch.Export.image.toDrive(
            image=img_sel,
            description=desc,
            folder='GEE',
            fileNamePrefix=desc,
            scale=30,
            maxPixels=1e13
        )
        task.start()
        logger.info(f"Started export task for {desc}")

if __name__ == "__main__":
    # log in earthengine using CLI before running this script
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="TreeMap2022", choices=DATASET_NAMES.keys())
    # for TreeMap, the bands are 'STANDHT', 'FLDTYPCD' # for ISDASOIL, the bands are 'texture_0_20' # for GFC, the bands are 'lossyear'
    parser.add_argument("--bands", nargs='+', type=str, default=None) 
    args = parser.parse_args()
    main(args)
    # use ``earthengine task list`` to check the task status
        