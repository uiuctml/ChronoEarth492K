<div align="center">

<h1>ChronoEarth-492K</h1>
<h3>A Large Scale and Long Horizon Spatiotemporal Hyperspectral Earth Observation Dataset and Benchmark</h3>

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://uiuctml.github.io/ChronoEarth492K)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv)](https://arxiv.org/abs/2605.15666)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Dataset-purple?logo=huggingface&logoColor=yellow)](https://huggingface.co/GFM-Bench/datasets)
[![GitHub](https://img.shields.io/badge/GitHub-ChronoEarth492K-green?logo=github&logoColor=white)](https://github.com/uiuctml/ChronoEarth492K)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

</div>

**ChronoEarth-492K** is a large-scale, temporally-rich hyperspectral remote sensing dataset built from EO-1 Hyperion imagery. It spans **492K+ locations** across **9 global regions**, covering **155 hyperspectral bands** from 447 nm to 2365 nm, with multiple revisits per location for temporal analysis.

Authors:
[Haozhe Si](https://ehzoahis.github.io/),
Yuxuan Wan,
Yuqing Wang,
[Minh Do](https://minhdo.ece.illinois.edu/),
[Han Zhao](https://hanzhaoml.github.io/).

---

## Dataset at a Glance

| Property | Value |
|---|---|
| **Locations** | 492K+ spatial locations |
| **Spectral Bands** | 155 hyperspectral bands (447–2365 nm) |
| **Spatial Resolution** | 30 m / pixel |
| **Temporal Coverage** | Multi-year, multiple revisits per location |
| **Global Regions** | 9 (AC, AF, EA, EU, LA, NA, OC, SEA, SWA) |
| **Sensor** | EO-1 Hyperion |

### Spectral Groups

| Group | Bands | Wavelength Range | # Channels |
|---|---|---|---|
| VNIR | B010–B057 | 447–925 nm | 48 |
| SWIR1 | B081–B097 | 952–1114 nm | 17 |
| SWIR2 | B101–B119 | 1154–1336 nm | 19 |
| SWIR3 | B134–B164 | 1487–1790 nm | 31 |
| SWIR4 | B182–B221 | 1971–2365 nm | 40 |
| **All** | | **447–2365 nm** | **155** |

### Benchmark Datasets

ChronoEarth-492K includes benchmark labels for six geospatial tasks, supporting both static and temporal evaluation:

| Dataset | Task | Region | Classes |
|---|---|---|---|
| **NLCD** | Land cover segmentation | North America | 16 |
| **CDL** | Crop type mapping | North America | 257 |
| **CORINE** | Land cover segmentation | Europe | 44 |
| **CLCD** | Land cover segmentation | China | 9 |
| **GFC** | Forest cover change detection | Global | 2 |
| **ISDASoil** | Soil property prediction | Africa | Continuous |

---

## Getting Started

### Installation

```bash
git clone https://github.com/uiuctml/ChronoEarth492K.git
cd ChronoEarth492K
pip install -r requirements.txt
```

### Download the Dataset

The dataset is hosted on Hugging Face. Download it using the provided utility:

```bash
python dataset_tools/download_data_hf.py --cache_dir /path/to/data
```

Or load it directly with the `datasets` library:

```python
from datasets import load_dataset
ds = load_dataset("GFM-Bench/ChronoEarth492K", split="train")
```

### Basic Usage

**Static dataset (single image per location):**

```python
from ChronoEarth import ChronoEarth

dataset = ChronoEarth(
    split='train',
    cache_dir='/path/to/data',
    regions=['NA', 'EU'],             # subset of global regions
    channel_groups=['VNIR', 'SWIR1'], # subset of spectral groups
)

sample = dataset[0]
# sample['optical']              -> np.ndarray (C, H, W)
# sample['optical_channel_wv']   -> list of wavelengths [nm]
# sample['spatial_resolution']   -> 30 (meters)
```

**Temporal dataset (time-series per location):**

```python
from ChronoEarth import TemporalChronoEarth

dataset = TemporalChronoEarth(
    split='train',
    cache_dir='/path/to/data',
    regions='NA',
    channel_groups='VNIR',
    num_frames=8,   # max frames to sample per location (-1 for all)
    frames_lb=3,    # only include locations with ≥3 timestamps
)

sample = dataset[0]
# sample['optical']    -> np.ndarray (T, C, H, W)
# sample['timestamp']  -> np.ndarray (T,)
# sample['num_frames'] -> int
```

**Benchmark datasets:**

```python
from ChronoEarth.benchmarks import StaticTask, TemporalTask

# Static benchmark
dataset = StaticTask(
    dataset_name='NLCD',
    split='train',
    metadata_path='/path/to/benchmark_labels/NLCD/metadata_static.parquet',
    image_dir='/path/to/dataset',
    label_dir='/path/to/benchmark_labels/NLCD/labels',
    BANDS=[30, 24, 16],  # RGB from VNIR
)
```

---

## Dataset Structure

```
ChronoEarth-492K/
 ├── metadata.parquet              # global index (location_uid, region, timestamp, image_path)
 ├── dataset/
 │     └── <UTM_zone>/
 │           └── <UID>/
 │                 ├── <UID>_<timestamp>.TIF
 │                 └── ...
 └── benchmark_labels/
       └── <dataset_name>/
             ├── *_metadata_static.parquet
             ├── *_metadata_lh.parquet   # long-horizon temporal
             └── labels/
                   └── <UID>/
                         ├── <dataset_name>_<year>.tif
                         └── ...
```

See [`docs/Dataset_Structure.md`](docs/Dataset_Structure.md) for the full schema.

---

## Pretraining

Scripts for pretraining geospatial foundation models on ChronoEarth-492K are provided under `pretraining/`. Both static and temporal pretraining are supported:

```bash
# Static pretraining
bash scripts/pretraining/launch_static.sh

# Temporal pretraining
bash scripts/pretraining/launch_temporal.sh
```

## Evaluation

Fine-tuning and evaluation on the benchmark datasets:

```bash
# Fine-tuning sweep
python evaluation/sweep_finetune.py \
    --dataset NLCD \
    --root_dir /path/to/checkpoints \
    --modal optical

# Temporal fine-tuning
python evaluation/launchers/launch_temporal_finetune_sweep.py \
    --dataset GFC \
    --root_dir /path/to/checkpoints
```

---

## Citation

If you find ChronoEarth-492K useful in your research, please cite our paper:

```bibtex
@misc{si2026chronoearth492klargescalelong,
      title={ChronoEarth-492K: A Large Scale and Long Horizon Spatiotemporal Hyperspectral Earth Observation Dataset and Benchmark}, 
      author={Haozhe Si and Yuxuan Wan and Yuqing Wang and Minh Do and Han Zhao},
      year={2026},
      eprint={2605.15666},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.15666}, 
}
```

---

## Contact

[Haozhe Si](mailto:haozhes3@illinois.edu) · [Han Zhao](mailto:hanzhao@illinois.edu)
