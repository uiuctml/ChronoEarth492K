from pathlib import Path
import math
import numpy as np
import pyogrio
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.transform import from_origin
from rasterio.crs import CRS

# --- inputs ---
SHP = "/home/haozhesi/EO1H-313K/data/archived/EuroCrop/FR_2018/FR_2018/FR_2018_EC21.shp"
OUT_DIR = Path("/home/haozhesi/EO1H-313K/data/FRCrop/FRCrop_raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RES = 30.0                               # meters
TILE_PX = 4096                           # tile size in pixels (≈ 123 km at 30 m)
LABEL = "EC_hcat_c"
DTYPE = "uint32"
NODATA = 0

# 1) Get layer extent and CRS
info = pyogrio.read_info(SHP)
src_crs = CRS.from_user_input(info["crs"])
minx, miny, maxx, maxy = info["total_bounds"]    # already in layer CRS

# 2) Snap extent to 30 m grid to avoid seams
def snap_down(x, step): return math.floor(x / step) * step
def snap_up(x, step):   return math.ceil(x / step) * step

minx = snap_down(minx, RES)
miny = snap_down(miny, RES)
maxx = snap_up(maxx, RES)
maxy = snap_up(maxy, RES)

# 3) Derive tile size in map units
tile_w = TILE_PX * RES
tile_h = TILE_PX * RES

# 4) Loop over tiles
x = minx
col = 0
while x < maxx:
    y = miny
    row = 0
    while y < maxy:
        # current tile bbox in layer CRS
        bx0, by0 = x, y
        bx1, by1 = min(x + tile_w, maxx), min(y + tile_h, maxy)

        # read only intersecting features (very fast)
        sub = pyogrio.read_dataframe(
            SHP,
            bbox=(bx0, by0, bx1, by1),
            columns=[LABEL, "geometry"],
            use_arrow=True,
        )

        # build this tile's transform (north-up)
        # from_origin(left, top, xres, yres)
        transform = from_origin(bx0, by1, RES, RES)
        height = int((by1 - by0) / RES)
        width  = int((bx1 - bx0) / RES)

        out = OUT_DIR / f"FR_2018_EC21_30m_r{row:04d}_c{col:04d}.tif"

        if len(sub) == 0:
            # empty tile → write zeros quickly
            with rasterio.open(
                out, "w",
                driver="GTiff", height=height, width=width, count=1,
                dtype=DTYPE, crs=src_crs, transform=transform,
                nodata=NODATA, tiled=True, compress="lzw", BIGTIFF="IF_NEEDED"
            ) as dst:
                dst.write(np.zeros((height, width), dtype=DTYPE), 1)
        else:
            # (optional) precise clip to bbox to reduce raster work
            sub = gpd.GeoDataFrame(sub, geometry="geometry", crs=src_crs)
            # prepare (geom, value) pairs and burn
            shapes = ((geom, int(val)) for geom, val in zip(sub.geometry, sub[LABEL]))
            arr = features.rasterize(
                shapes=shapes,
                out_shape=(height, width),
                transform=transform,
                fill=NODATA,
                dtype=DTYPE,
                # all_touched=True,  # enable if you prefer inclusive edges
            )
            with rasterio.open(
                out, "w",
                driver="GTiff", height=height, width=width, count=1,
                dtype=DTYPE, crs=src_crs, transform=transform,
                nodata=NODATA, tiled=True, compress="lzw", BIGTIFF="IF_NEEDED"
            ) as dst:
                dst.write(arr, 1)

        row += 1
        y = by1
    col += 1
    x = bx1

# (Optional) Build a mosaic later:
# gdalbuildvrt FR_2018_EC21_30m.vrt /home/.../tiles_30m/*.tif
# gdal_translate FR_2018_EC21_30m.vrt FR_2018_EC21_30m.tif -co TILED=YES -co COMPRESS=LZW -co BIGTIFF=IF_NEEDED