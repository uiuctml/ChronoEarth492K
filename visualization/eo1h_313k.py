import numpy as np

from typing import List
from torch.utils.data import Dataset
from datasets import load_dataset, concatenate_datasets

WV_MAX = 2365.20
WV_MIN = 447.17

# EO1H-313K data statistics
NUM_CHANNELS = {
    "SWIR1": 17,
    "SWIR2": 19,
    "SWIR3": 31,
    "SWIR4": 40,
    "VNIR": 48
}

SWIR1_MEAN = [1854.6320, 2651.0768, 3295.3672, 3837.7938, 4209.5321, 4149.5452,
              4165.5854, 4141.0900, 4028.4024, 3970.3287, 3857.1756, 3714.6392,
              3613.5317, 3441.5520, 3018.4122, 2357.3898, 1181.8382]
SWIR1_STD = [1258.2504, 1645.1517, 1997.6949, 2301.8653, 2526.2403, 2482.4821,
             2477.0938, 2452.5793, 2385.4237, 2343.1566, 2287.0967, 2200.2029,
             2129.3652, 2023.6333, 1775.2120, 1408.2428,  804.2104]

SWIR2_MEAN = [1356.8760, 2051.9419, 2266.8682, 2254.8853, 2294.9750, 2281.2388,
              2445.0167, 2579.4688, 2662.2673, 2613.0851, 2381.9764, 2061.6785,
              2290.3143, 2376.2906, 2200.1077, 1880.4995, 1574.2302, 1174.7962,
              841.9733]
SWIR2_STD = [927.3823, 1306.2522, 1429.1399, 1435.8325, 1478.9025, 1468.1460,
             1554.9112, 1637.1585, 1697.1882, 1664.6492, 1487.5500, 1284.1737,
             1433.8444, 1482.2878, 1372.0322, 1180.8281, 1053.6969,  775.7937,
             592.4241]

SWIR3_MEAN = [565.8670, 884.8449, 1068.1087, 1201.4064, 1296.9499, 1353.2338,
              1379.4769, 1362.9483, 1249.3019, 1224.5416, 1279.4565, 1224.2350,
              1216.7346, 1286.1854, 1291.7444, 1220.4995, 1207.0306, 1198.6247,
              1164.5403, 1148.7208, 1115.2409, 1068.0144, 1013.4791,  955.1148,
              884.8343,  823.1042,  794.1952,  719.7178,  589.2938,  435.0857,
              297.6542]
SWIR3_STD = [527.0507, 767.6636, 911.8197, 1018.0433, 1087.3382, 1113.3161, 
             1122.2665, 096.6123, 994.2635, 971.5529, 1004.3463, 955.0387,
             966.3975, 984.3837, 981.1252, 923.0221, 910.4236, 904.0315,
             880.5475, 871.6562, 849.0370, 815.2917, 778.1466, 737.3997,
             685.8484, 639.0191, 618.5641, 567.8635, 465.2104, 348.4695,
             248.6622]

SWIR4_MEAN = [186.3175, 285.8867, 243.5926,  57.3199,  69.7652, 223.0510, 373.1152,
              342.3601, 227.5900, 251.0881, 312.7170, 372.4590, 393.5948, 408.8218, 
              405.9004, 397.9031, 414.1036, 409.5424, 378.3310, 345.1416, 344.2391, 
              342.6009, 329.3066, 306.9529, 326.4912, 329.1512, 316.5052, 290.5734, 
              273.1225, 267.0219, 262.8438, 244.7820, 219.3891, 232.0303, 208.0891,
              210.6692, 194.8225, 162.5772, 166.2536, 158.5425]
SWIR4_STD = [201.0392, 291.0891, 247.5008,  72.6323,  86.5849, 220.8949, 360.5947,
             339.2334, 233.6322, 247.4894, 304.8593, 368.7983, 388.0439, 395.5279,
             390.9397, 389.8140, 515.1792, 384.0314, 351.5467, 328.6297, 325.2252,
             307.9828, 293.0391, 277.8086, 293.8176, 292.4966, 283.5418, 271.0783,
             259.9661, 251.8127, 250.0389, 235.8894, 214.0307, 226.6427, 203.7511,
             205.8618, 191.0648, 161.1038, 164.1999, 155.6791]

VNIR_MEAN = [2612.4793, 2700.8952, 2766.7690, 2905.3839, 2712.6726, 2728.7971,
             2767.7316, 2645.7654, 2755.5231, 2772.2124, 2820.1131, 2870.5478,
             2836.7768, 2838.0846, 2914.9838, 2895.9597, 2984.5386, 2909.7780,
             2830.5010, 2855.6915, 2867.4339, 2735.4647, 2850.2696, 2829.9077,
             2608.8445, 2684.4396, 2725.1659, 2466.8938, 2792.1863, 3118.3264,
             2993.6927, 2056.2041, 3026.2286, 3072.0198, 2910.7983, 2829.0360,
             2460.8710, 2311.3275, 2479.7012, 2724.6049, 2721.3460, 2683.3118,
             2613.8901, 2494.9851, 2122.4396, 1761.8558, 1612.8144, 1596.7610]
VNIR_STD = [1343.1251, 1439.1274, 1524.0445, 1651.5318, 1591.0651, 1651.6324,
            1718.8263, 1673.2682, 1760.3685, 1795.3709, 1872.1368, 1962.2828,
            1990.5647, 2081.2126, 2189.7596, 2194.2596, 2313.1158, 2282.8147,
            2246.7044, 2290.7178, 2325.8293, 2243.9780, 2374.7051, 2379.6259,
            2176.0635, 2159.5033, 1981.6764, 1699.5278, 1816.7910, 1959.9704,
            1858.0160, 1267.0520, 1852.2551, 1872.2114, 1767.0779, 1711.7271,
            1489.9334, 1397.4679, 1486.0524, 1625.1747, 1617.5715, 1590.4289,
            1545.9595, 1475.2877, 1258.2956, 1056.8230,  967.8985,  951.3343]

class EO1H313K(Dataset):
    spatial_resolution = 30 
    metadata = { 
        "SWIR1": {
            "bands": ['B81', 'B82', 'B83', 'B84', 'B85', 'B86', 'B87', 'B88', 'B89', 
                      'B90', 'B91', 'B92', 'B93', 'B94', 'B95', 'B96', 'B97'],
            "channel_wv": [952.82, 962.91, 972.99, 983.08, 993.17, 1003.30, 1013.30, 1023.40, 
                           1033.49, 1043.59, 1053.69, 1063.79, 1073.89, 1083.99, 1094.09, 1104.19, 1114.19],
            "mean": SWIR1_MEAN,
            "std": SWIR1_STD
        },
        "SWIR2": {
            "bands": ['B101', 'B102', 'B103', 'B104', 'B105', 'B106', 'B107', 'B108', 
                      'B109', 'B110', 'B111', 'B112', 'B113', 'B114', 'B115', 'B116', 'B117', 'B118', 'B119'],
            "channel_wv": [1154.58, 1164.68, 1174.77, 1184.87, 1194.97, 1205.07, 1215.17, 1225.17, 1235.27, 
                           1245.36, 1255.46, 1265.56, 1275.66, 1285.76, 1295.86, 1305.96, 1316.05, 1326.05, 1336.15],
            "mean": SWIR2_MEAN,
            "std": SWIR2_STD
        },
        "SWIR3": {
            "bands": ['B134', 'B135', 'B136', 'B137', 'B138', 'B139', 'B140', 'B141', 'B142', 'B143', 'B144', 
                      'B145', 'B146', 'B147', 'B148', 'B149', 'B150', 'B151', 'B152', 'B153', 'B154', 'B155', 
                      'B156', 'B157', 'B158', 'B159', 'B160', 'B161', 'B162', 'B163', 'B164'],
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
            "channel_wv": [1971.76, 1981.86, 1991.96, 2002.06, 2012.15, 2022.25, 2032.35, 2042.45, 2052.45, 2062.55, 
                           2072.65, 2082.75, 2092.84, 2102.94, 2113.04, 2123.14, 2133.24, 2143.34, 2153.34, 2163.43, 
                           2173.53, 2183.63, 2193.73, 2203.83, 2213.93, 2224.03, 2234.12, 2244.22, 2254.22, 2264.32, 
                           2274.42, 2284.52, 2294.61, 2304.71, 2314.81, 2324.91, 2335.01, 2345.11, 2355.21, 2365.20],
            "mean": SWIR4_MEAN,
            "std": SWIR4_STD
        },
        "VNIR": {
            "bands": ['B10', 'B11', 'B12', 'B13', 'B14', 'B15', 'B16', 'B17', 'B18', 
                      'B19', 'B20', 'B21', 'B22', 'B23', 'B24', 'B25', 'B26', 'B27', 
                      'B28', 'B29', 'B30', 'B31', 'B32', 'B33', 'B34', 'B35', 'B36', 
                      'B37', 'B38', 'B39', 'B40', 'B41', 'B42', 'B43', 'B44', 'B45', 
                      'B46', 'B47', 'B48', 'B49', 'B50', 'B51', 'B52', 'B53', 'B54', 
                      'B55', 'B56', 'B57'],
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

    def __init__(self, split: str, cache_dir: str, regions: List[str] | str, channel_groups: List[str] | str):
        assert split == 'train', f"please use train split"
        self.cache_dir = cache_dir
        self.regions = regions if isinstance(regions, list) else [regions]
        self.channel_groups = channel_groups if isinstance(channel_groups, list) else [channel_groups]
        self.size = 128

        self.dss = self._load_dataset(
            cache_dir=self.cache_dir,
            regions=self.regions,
            channel_groups=self.channel_groups,
            split=split
        )
        num_samples = len(self.dss[0])
        for i, ds in enumerate(self.dss):
            assert num_samples == len(ds), f"all dataset should have {num_samples} but got {len(ds)} in {channel_groups[i]}"

        self._metadata = self._get_metadata(channel_groups=channel_groups)

    def __len__(self):
        return len(self.dss[0])

    def _get_metadata(self, channel_groups: List[str]):
        channel_wv = []
        for channel_group in channel_groups:
            channel_wv += self.metadata[channel_group]["channel_wv"]

        return channel_wv

    def _concat_by_regions(self, cache_dir: str, channel_group: str, regions: List[str], split: str = "train"):
        """Concatenate multiple region configs for the same band group."""
        print(f"Loading {channel_group} data from regions: {regions}")
        parts = [
            load_dataset("GFM-Bench/EO1H-313K", name=f"{r}-{channel_group}", split=split, cache_dir=cache_dir)
            for r in regions
        ]
        return concatenate_datasets(parts)

    def _load_dataset(self, cache_dir: str, regions: List[str], channel_groups: List[str], split: str):
        dss = [
            self._concat_by_regions(
                cache_dir=cache_dir,
                channel_group=channel_group,
                regions=regions,
                split=split
            )
            for channel_group in channel_groups
        ] # A list of Dataset, with each being ds for one channel group of across regions

        return dss

    def __getitem__(self, idx):
        optical = [ds[idx]['image'] for ds in self.dss]
        optical = np.concatenate(optical, axis=0)
        
        return {
            "optical": optical,
            "optical_channel_wv": self._metadata,
            "spatial_resolution": self.spatial_resolution
        }

class EO1H313K_full(EO1H313K):
    def __init__(self, split: str, cache_dir: str, regions: List[str] | str):
        super().__init__(
            split=split,
            cache_dir=cache_dir,
            regions=regions,
            channel_groups=['VNIR', 'SWIR1', 'SWIR2', 'SWIR3', 'SWIR4'] 
        )
    
    def _concat_by_regions(self, cache_dir: str, channel_group: str, regions: List[str], split: str = "train"):
        """Concatenate multiple region configs for the same band group."""
        print(f"Loading {channel_group} data from regions: {regions}")
        parts = [
            load_dataset("yuxuanw8/EO1H-313K-full", name=f"{r}", split=split, cache_dir=cache_dir)
            for r in regions
        ]
        return concatenate_datasets(parts)

    def _load_dataset(self, cache_dir: str, regions: List[str], channel_groups: List[str], split: str):
        dss =[
            self._concat_by_regions(
                cache_dir=cache_dir,
                channel_group="All channel groups",
                regions=regions,
                split=split
            )
        ]

        return dss
    
    def __getitem__(self, idx):
        optical = self.dss[0][idx]['image']
        
        return {
            "optical": optical,
            "optical_channel_wv": self._metadata,
            "spatial_resolution": self.spatial_resolution
        }

class EO1H313K_RGB(EO1H313K):
    def __init__(self, split: str, cache_dir: str, regions: List[str] | str):
        super().__init__(
            split=split,
            cache_dir=cache_dir,
            regions=regions,
            channel_groups=['VNIR'] 
        )
        target_bands = [30, 24, 16]
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
