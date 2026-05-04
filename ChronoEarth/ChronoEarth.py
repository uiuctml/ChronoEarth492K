import os
import sys
import time
import torch
import numpy as np

from typing import List
from torch.utils.data import Dataset, get_worker_info
from datasets import load_dataset, concatenate_datasets

import pandas as pd
import tifffile

ALL_REGIONS = ['AC', 'AF', 'EA', 'EU', 'LA', 'NA', 'OC', 'SEA', 'SWA']
ALL_CHANNEL_GROUPS = ['VNIR', 'SWIR1', 'SWIR2', 'SWIR3', 'SWIR4']

WV_MAX = 2365.20
WV_MIN = 447.17

# ChronoEarth data statistics
NUM_CHANNELS = {
    "SWIR1": 17,
    "SWIR2": 19,
    "SWIR3": 31,
    "SWIR4": 40,
    "VNIR": 48
}

SWIR1_MEAN = [1896.57772697, 2638.56970297, 3232.63944248, 3725.91059444, 4062.2612601, 
            3996.5943036, 4006.43602126, 3977.92931771, 3866.31213839, 3809.80514808, 
            3706.4297811, 3572.51271894, 3477.61301593, 3319.07221557, 2926.71960334, 
            2317.78161791, 1209.31383974]
SWIR1_STD = [1347.9701047, 1697.4028033, 2029.33114583, 2320.71457581, 2540.55401644, 
            2495.21180975, 2489.21577164, 2464.23858342, 2397.48525413, 2356.45503928, 
            2303.67495442, 2218.12110856, 2146.27794598, 2041.17563553, 1794.81446298, 
            1436.33700958, 859.62942701]

SWIR2_MEAN = [1360.71217641, 2003.37797075, 2194.58857349, 2176.14227142, 2206.633312, 
            2186.7904433, 2336.54788791, 2455.71966122, 2528.13542149, 2479.61891601, 
            2262.39914476, 1959.65152493, 2174.43347223, 2258.42217242, 2099.19507966, 
            1805.13670512, 1519.68562482, 1152.30283904, 842.93812909]
SWIR2_STD = [955.06771918, 1313.6284054, 1432.24620247, 1435.95084437, 1477.42301939, 
            1466.11423868, 1554.05394734, 1637.41455256, 1698.33642462, 1667.42027982, 
            1494.14131335, 1290.79122518, 1438.3145996, 1486.42262736, 1374.73089872, 
            1182.10268338, 1041.29176797, 781.38429309, 605.74248789]

SWIR3_MEAN = [547.56053166, 839.82038549, 1005.67149169, 1125.92373335, 1211.46185329, 
            1263.16586398, 1286.83477968, 1271.39622554, 1166.77322954, 1143.55661651, 
            1194.9912801, 1144.44761499, 1136.19188845, 1201.42334548, 1206.67255274, 
            1141.17491349, 1129.51476942, 1121.80322731, 1090.87455627, 1075.09658808, 
            1045.40193933, 1002.29414302, 952.2759967, 899.18971792, 834.32342978, 
            778.51161808, 752.31399883, 684.42275692, 565.751694, 424.75973439, 297.43936112]
SWIR3_STD = [526.16306102, 761.8053182, 904.415344, 1010.13699764, 1079.2353845, 
            1106.13222464, 1115.33810231, 1090.33771224, 989.70126096, 966.86291344, 
            1000.0171335, 951.98933714, 955.47655625, 981.59073287, 978.70675041, 
            921.28959068, 909.10820572, 902.52930844, 879.23476178, 869.44300589, 
            846.84586802, 812.75921337, 775.15337889, 734.61566213, 682.92492538, 
            637.4861037, 615.59005827, 563.18487628, 462.6377035, 349.0325118, 252.142348]

SWIR4_MEAN = [178.47999501, 271.14336916, 231.20706232, 55.62310245, 66.92127095, 
            210.32993422, 349.44678927, 320.73996208, 213.7775889, 236.14671413, 
            293.51710569, 348.64545148, 368.50811487, 382.77996379, 380.22141996, 
            372.62603138, 386.36051459, 383.61233678, 355.38220368, 324.12493745, 
            323.56794994, 323.13849541, 310.65395662, 289.30325042, 307.14417445, 
            309.52774477, 297.66774692, 273.50133436, 257.04341034, 251.53576908, 
            247.48388389, 230.51337299, 206.99096097, 218.36901604, 196.19504385, 
            198.48538361, 183.73376705, 153.92022232, 157.41694558, 150.55478027]
SWIR4_STD = [199.47026911, 288.35866265, 245.63199038, 70.61290307, 84.5159148, 
            219.03895348, 357.55257264, 336.85349611, 231.96218661, 245.52994827, 
            302.15438394, 365.29090395, 384.39363229, 391.71918818, 387.20333007, 
            385.97944674, 475.79640719, 379.98162616, 347.79933938, 325.22740495, 
            322.02924106, 305.16741082, 290.57692825, 275.50781299, 291.38991532, 
            290.05240003, 281.11132159, 268.72225761, 257.50656353, 249.18283304, 
            247.33727326, 233.54436985, 211.891135, 224.00097568, 201.06628892, 
            202.74485296, 187.97467477, 157.7714269, 160.8165076, 152.47282978]

VNIR_MEAN = [2641.97087503, 2735.06334724, 2804.26249984, 2947.06316199, 2751.64601612, 
            2768.17256915, 2806.49509711, 2678.96530112, 2785.56121863, 2796.81150404, 
            2839.83627572, 2882.37787646, 2838.97716429, 2832.65052655, 2903.80021216, 
            2876.13333612, 2951.77452771, 2873.21921899, 2793.9286001, 2813.84433927, 
            2828.15333628, 2693.57701745, 2801.19142647, 2779.85999852, 2567.14400901, 
            2637.96211547, 2681.39261876, 2443.78764478, 2743.78620736, 3047.17045307, 
            2931.04119453, 2017.17422121, 2944.75805206, 2997.53987527, 2843.72892171, 
            2764.65136131, 2424.86371763, 2273.95340839, 2426.8708351, 2655.07739785, 
            2648.91065944, 2609.07030482, 2539.64230928, 2424.5259912, 2078.72497804, 
            1733.53040949, 1589.85941905, 1571.42087585]
VNIR_STD = [1497.80450791, 1601.64230528, 1692.31137575, 1830.74130633, 1756.69818987, 
            1814.89777612, 1879.58111031, 1817.47313595, 1901.15975643, 1927.50043147, 
            2001.96177256, 2086.28490747, 2091.70192487, 2186.01105343, 2280.81438062, 
            2256.97425449, 2367.41858139, 2326.92551079, 2289.17604021, 2329.06729706, 
            2365.07900802, 2275.50769204, 2405.5915603, 2413.20805297, 2215.73704077, 
            2199.50060984, 2024.57441329, 1757.38226918, 1862.72825783, 1995.16442491, 
            1895.59454674, 1301.78349167, 1878.25088094, 1901.9616739, 1796.79197293, 
            1741.03230243, 1528.38580434, 1432.62840953, 1515.82047988, 1652.02616713, 
            1643.56402607, 1615.00829696, 1569.05806418, 1497.46544681, 1285.39861311, 
            1083.86908308, 995.9395129, 977.72673921]

RGB_BANDS = [30, 24, 16]
VNIR_BANDS = list(range(10, 58))
SWIR1_BANDS = list(range(81, 98))
SWIR2_BANDS = list(range(101, 120))
SWIR3_BANDS = list(range(134, 165))
SWIR4_BANDS = list(range(182, 222))
ALL_BANDS = VNIR_BANDS + SWIR1_BANDS + SWIR2_BANDS + SWIR3_BANDS + SWIR4_BANDS

ALL_MEANS = VNIR_MEAN + SWIR1_MEAN + SWIR2_MEAN + SWIR3_MEAN + SWIR4_MEAN
ALL_STDS = VNIR_STD + SWIR1_STD + SWIR2_STD + SWIR3_STD + SWIR4_STD

channel_metadata = { 
    "SWIR1": {
        "bands": ['B081', 'B082', 'B083', 'B084', 'B085', 'B086', 'B087', 'B088', 'B089', 
                'B090', 'B091', 'B092', 'B093', 'B094', 'B095', 'B096', 'B097'],
        "band_index": SWIR1_BANDS,
        "channel_wv": [952.82, 962.91, 972.99, 983.08, 993.17, 1003.30, 1013.30, 1023.40, 
                    1033.49, 1043.59, 1053.69, 1063.79, 1073.89, 1083.99, 1094.09, 1104.19, 1114.19],
        "mean": SWIR1_MEAN,
        "std": SWIR1_STD
    },
    "SWIR2": {
        "bands": ['B101', 'B102', 'B103', 'B104', 'B105', 'B106', 'B107', 'B108', 
                'B109', 'B110', 'B111', 'B112', 'B113', 'B114', 'B115', 'B116', 'B117', 'B118', 'B119'],
        "band_index": SWIR2_BANDS,
        "channel_wv": [1154.58, 1164.68, 1174.77, 1184.87, 1194.97, 1205.07, 1215.17, 1225.17, 1235.27, 
                    1245.36, 1255.46, 1265.56, 1275.66, 1285.76, 1295.86, 1305.96, 1316.05, 1326.05, 1336.15],
        "mean": SWIR2_MEAN,
        "std": SWIR2_STD
    },
    "SWIR3": {
        "bands": ['B134', 'B135', 'B136', 'B137', 'B138', 'B139', 'B140', 'B141', 'B142', 'B143', 'B144', 
                'B145', 'B146', 'B147', 'B148', 'B149', 'B150', 'B151', 'B152', 'B153', 'B154', 'B155', 
                'B156', 'B157', 'B158', 'B159', 'B160', 'B161', 'B162', 'B163', 'B164'],
        "band_index": SWIR3_BANDS,
        "channel_wv": [1487.53, 1497.63, 1507.73, 1517.83, 1527.92, 1537.92, 1548.02, 1558.12, 1568.22, 1578.32, 
                    1588.42, 1598.51, 1608.61, 1618.71, 1628.81, 1638.81, 1648.90, 1659.00, 1669.10, 1679.20, 
                    1689.30, 1699.40, 1709.50, 1719.60, 1729.70, 1739.70, 1749.79, 1759.89, 1769.99, 1780.09, 
                    1790.19],
        "mean": SWIR3_MEAN,
        "std": SWIR3_STD
    },
    "SWIR4": {
        "bands": ['B182', 'B183', 'B184', 'B185', 'B186', 'B187', 'B188', 'B189', 'B190', 'B191', 'B192', 
                'B193', 'B194', 'B195', 'B196', 'B197', 'B198', 'B199', 'B200', 'B201', 'B202', 'B203', 
                'B204', 'B205', 'B206', 'B207', 'B208', 'B209', 'B210', 'B211', 'B212', 'B213', 'B214', 
                'B215', 'B216', 'B217', 'B218', 'B219', 'B220', 'B221'],
        "band_index": SWIR4_BANDS,
        "channel_wv": [1971.76, 1981.86, 1991.96, 2002.06, 2012.15, 2022.25, 2032.35, 2042.45, 2052.45, 2062.55, 
                    2072.65, 2082.75, 2092.84, 2102.94, 2113.04, 2123.14, 2133.24, 2143.34, 2153.34, 2163.43, 
                    2173.53, 2183.63, 2193.73, 2203.83, 2213.93, 2224.03, 2234.12, 2244.22, 2254.22, 2264.32, 
                    2274.42, 2284.52, 2294.61, 2304.71, 2314.81, 2324.91, 2335.01, 2345.11, 2355.21, 2365.20],
        "mean": SWIR4_MEAN,
        "std": SWIR4_STD
    },
    "VNIR": {
        "bands": ['B010', 'B011', 'B012', 'B013', 'B014', 'B015', 'B016', 'B017', 'B018', 
                'B019', 'B020', 'B021', 'B022', 'B023', 'B024', 'B025', 'B026', 'B027', 
                'B028', 'B029', 'B030', 'B031', 'B032', 'B033', 'B034', 'B035', 'B036', 
                'B037', 'B038', 'B039', 'B040', 'B041', 'B042', 'B043', 'B044', 'B045', 
                'B046', 'B047', 'B048', 'B049', 'B050', 'B051', 'B052', 'B053', 'B054', 
                'B055', 'B056', 'B057'],
        "band_index": VNIR_BANDS,
        "channel_wv": [447.170, 457.340, 467.520, 477.690, 487.870, 498.040, 508.220, 518.390,
                    528.570, 538.740, 548.920, 559.090, 569.270, 579.450, 589.620, 599.800,
                    609.970, 620.150, 630.320, 640.500, 650.670, 660.850, 671.020, 681.200,
                    691.370, 701.550, 711.720, 721.900, 732.070, 742.250, 752.420, 762.600,
                    772.770, 782.950, 793.120, 803.300, 813.470, 823.650, 833.820, 844.000,
                    854.170, 864.350, 874.520, 884.700, 894.870, 905.050, 915.220, 925.410],
        "mean": VNIR_MEAN,
        "std": VNIR_STD
    },
}

       
class ChronoEarth(Dataset):
    spatial_resolution = 30 
    metadata = channel_metadata

    def __init__(self, split: str, cache_dir: str, regions: List[str] | str = ALL_REGIONS, channel_groups: List[str] | str = ALL_CHANNEL_GROUPS, transform=None):
        assert split == 'train', f"please use train split"
        self.cache_dir = cache_dir
        self.regions = regions if isinstance(regions, list) else [regions]
        self.channel_groups = channel_groups if isinstance(channel_groups, list) else [channel_groups]
        self.size = 128
        self.BANDS = np.concatenate([channel_metadata[channel_group]['bands'] for channel_group in channel_groups])
        self.BANDS.sort()
        self.transform = transform

        try:
            self.dss = pd.read_parquet(os.path.join(cache_dir, 'metadata.parquet'))
        except:
            self.dss = pd.read_csv(os.path.join(cache_dir, 'metadata.csv'))

        self.dss = self.dss.loc[self.dss['region'].isin(self.regions)]

        self._metadata = self._get_metadata(channel_groups=channel_groups)

    def __len__(self):
        return len(self.dss)

    def _get_metadata(self, channel_groups: List[str]):
        channel_wv = []
        for channel_group in channel_groups:
            channel_wv += self.metadata[channel_group]["channel_wv"]

        return channel_wv
    
    def _load_image(self, image_path: str):
        try:
            optical_path = os.path.join(self.cache_dir, image_path)
            optical = tifffile.imread(optical_path).astype(np.float32)
            optical = np.transpose(optical, (2, 0, 1))
        except:
            # read channel by channel
            optical = []
            for band in self.BANDS:
                optical_path = os.path.join(self.cache_dir, image_path.format(BAND=band))
                optical.append(tifffile.imread(optical_path).astype(np.float32))
            optical = np.stack(optical, axis=0)
        optical = np.nan_to_num(optical, nan=0.0)
        return optical

    def __getitem__(self, idx):
        optical_path = os.path.join(self.cache_dir, self.dss.iloc[idx]['image_path'])
        optical = self._load_image(optical_path)
        
        if self.transform is not None:
            optical, spatial_resolution = self.transform(
                optical=optical,
                spatial_resolution=self.spatial_resolution
            )
        spatial_resolution = self.spatial_resolution
        
        return {
            "optical": optical,
            "optical_channel_wv": self._metadata,
            "spatial_resolution": spatial_resolution
        }
        
class TemporalChronoEarth(ChronoEarth):
    def __init__(self, split: str, cache_dir: str, regions: List[str] | str = ALL_REGIONS, channel_groups: List[str] | str = ALL_CHANNEL_GROUPS,
                 num_frames: int = -1, frames_lb: int = 1, transform=None, slow_sample_threshold: float = 0.0):
        super().__init__(split, cache_dir, regions, channel_groups)
        # filter the dss by the frames_lb
        temporal_counts = self.dss["location_uid"].value_counts()
        dense_location_uids = temporal_counts[temporal_counts >= frames_lb].index
        self.metadata = self.dss.loc[self.dss["location_uid"].isin(dense_location_uids)]
        self.metadata_groups = list(self.metadata.groupby("location_uid"))
        self.num_frames = num_frames
        self.frames_lb = frames_lb
        self.frame_lengths = self._get_frame_lengths()
        if self.num_frames > 0:
            self.frame_lengths = np.minimum(self.frame_lengths, self.num_frames)
        self.transform = transform
        self.slow_sample_threshold = slow_sample_threshold
        
    def __len__(self):
        return len(self.metadata_groups)
    
    def _get_frame_lengths(self):
        frame_lengths = []
        for _, df in self.metadata_groups:
            timestamps = np.array(df['timestamp'].tolist())
            frame_lengths.append(len(timestamps))
        return np.array(frame_lengths)
        
    def __getitem__(self, idx):
        start_time = time.perf_counter()
        location_uid, df = self.metadata_groups[idx]
        df = df.sort_values(by='timestamp', ascending=True).reset_index(drop=True)
        img_dirs = df['image_path'].tolist()
        timestamps = np.array(df['timestamp'].tolist())
        n_timestamps = len(timestamps)

        if self.num_frames > 0 and n_timestamps > self.num_frames:
            kept_idx = np.random.choice(n_timestamps, self.num_frames, replace=False)
            kept_idx = np.sort(kept_idx)
            img_dirs = [img_dirs[i] for i in kept_idx]
            timestamps = timestamps[kept_idx]
            n_timestamps = self.num_frames

        opticals = []
        for img_dir in img_dirs:
            frame_start_time = time.perf_counter()
            optical = self._load_image(img_dir)
            frame_elapsed = time.perf_counter() - frame_start_time
            if self.slow_sample_threshold and frame_elapsed >= self.slow_sample_threshold:
                worker = get_worker_info()
                worker_id = "main" if worker is None else worker.id
                print(
                    f"[TemporalChronoEarth] slow frame read: worker={worker_id} "
                    f"idx={idx} location_uid={location_uid} image_path={img_dir} "
                    f"elapsed={frame_elapsed:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )
            opticals.append(optical)
        opticals = np.stack(opticals, axis=0)
            
        if self.transform is not None:
            opticals, spatial_resolution = self.transform(
                optical=opticals,
                spatial_resolution=self.spatial_resolution
            )
        spatial_resolution = self.spatial_resolution

        elapsed = time.perf_counter() - start_time
        if self.slow_sample_threshold and elapsed >= self.slow_sample_threshold:
            worker = get_worker_info()
            worker_id = "main" if worker is None else worker.id
            print(
                f"[TemporalChronoEarth] slow sample: worker={worker_id} "
                f"idx={idx} location_uid={location_uid} frames={n_timestamps} "
                f"elapsed={elapsed:.1f}s",
                file=sys.stderr,
                flush=True,
            )
        
        return {
            "optical": opticals,
            "timestamp": timestamps,
            "num_frames": n_timestamps,
            "optical_channel_wv": self._metadata,
            "spatial_resolution": spatial_resolution
        }

class ChronoEarth_dev(ChronoEarth):
    def __getitem__(self, idx):
        # optical = [ds[idx]['image'] for ds in self.dss]
        frame_id = self.dss[0][idx]["frame_id"]
        geo = self.dss[0][idx]["geo"]
        # optical = np.concatenate(optical, axis=0)
        
        return {
            # "optical": optical,
            "frame_id": frame_id,
            "geo": geo
        }
        
class ChronoEarth_RGB(ChronoEarth):
    def __init__(self, split: str, cache_dir: str, regions: List[str] | str):
        super().__init__(
            split=split,
            cache_dir=cache_dir,
            regions=regions,
            channel_groups=['VNIR'] 
        )
        target_bands = RGB_BANDS
        self.idx = [self.metadata['VNIR']['bands'].index(f'B{b}') for b in target_bands] 
    
    def __getitem__(self, idx):
        optical = self.dss[0][idx]['image']
        optical = np.array(optical)
        optical = optical[self.idx]  # Select R, G, B channels from VNIR
        
        return {
            "optical": optical,
            "optical_channel_wv": [self._metadata[29], self._metadata[19], self._metadata[9]],
            "spatial_resolution": self.spatial_resolution,
            "geo": self.dss[0][idx]['geo']
        }
 
