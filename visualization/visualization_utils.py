import rasterio
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from tqdm import tqdm
from pathlib import Path
# from eo1h_313k import EO1H313K_RGB
import glob
from visualization.color_maps import *
import math

YEARS = range(2001, 2018)

VNIR_BANDS = ['B10', 'B11', 'B12', 'B13', 'B14', 'B15', 'B16', 'B17', 'B18', 
            'B19', 'B20', 'B21', 'B22', 'B23', 'B24', 'B25', 'B26', 'B27', 
            'B28', 'B29', 'B30', 'B31', 'B32', 'B33', 'B34', 'B35', 'B36', 
            'B37', 'B38', 'B39', 'B40', 'B41', 'B42', 'B43', 'B44', 'B45', 
            'B46', 'B47', 'B48', 'B49', 'B50', 'B51', 'B52', 'B53', 'B54', 
            'B55', 'B56', 'B57']
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

VNIR_STATS = {band: (mean, std) for band, mean, std in zip(VNIR_BANDS, VNIR_MEAN, VNIR_STD)}

def vis_image(img, caption=None, bbox_anchor=(0.5, -0.2)):
    img = np.transpose(img, (1, 2, 0))  # (H, W, 3)

    # Normalize using percentile clipping (robust to outliers)
    if img.dtype != np.uint8:
        # Clip to 2nd and 98th percentiles for each channel
        img_normalized = np.zeros_like(img, dtype=np.float32)
        for i in range(img.shape[2]):
            channel = img[:, :, i]
            p_low = np.percentile(channel, 2)
            p_high = np.percentile(channel, 98)
            clipped = np.clip(channel, p_low, p_high)
            img_normalized[:, :, i] = (clipped - p_low) / (p_high - p_low)
        img = img_normalized

    plt.figure(figsize=(4,4))
    plt.imshow(img)
    plt.axis("off")
    if caption is not None:
        plt.title(caption)
    plt.show()
    
def vis_images(imgs, timestamps=None, captions=None, bbox_anchor=(0.5, -0.2)):
    n_imgs = len(imgs)
    n_cols = n_imgs
    n_rows = 1
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4+2))
    axs = axs.flatten()
    for i in range(n_imgs):
        img = imgs[i]
        img = _prepare_img(img)
        axs[i].imshow(img)
        axs[i].axis("off")
        if timestamps is not None:
            year = timestamps[i] // 365 + 2000
            day = timestamps[i] % 365
            axs[i].set_title(f"Year: {year}, Day: {day}")
    if captions is not None:
        fig.text(
            *bbox_anchor, captions,
            ha="center", va="bottom",
        )
    plt.show()

def vis_label(label_path, cdl_colors=CDL, return_value=False, process_label=None, bbox_anchor=(0.5, -0.2), label_idx=1):
    with rasterio.open(label_path) as src:
        label = src.read(label_idx)
    label = label.reshape(1, label.shape[0], label.shape[1])
    if process_label is not None:
        label = process_label(label)
        print(label.shape)
    label, legend_items = _prepare_label(label, colors=cdl_colors)
    fig = plt.figure(figsize=(4, 4))
    plt.imshow(label)
    patches = [mpatches.Patch(color=color, label=name) 
               for name, color in legend_items]
    
    # Add legend below both images with more space
    fig.legend(handles=patches, loc='lower center', 
               ncol=min(len(patches), 4), frameon=True, 
               fontsize=10, bbox_to_anchor=bbox_anchor)
    plt.axis("off")
    plt.show()
    if return_value:
        return label, legend_items
    
def vis_regression_label(label_path, return_value=False, bbox_anchor=(0.5, -0.2), label_idx=1):
    with rasterio.open(label_path) as src:
        label = src.read(label_idx)
    fig = plt.figure(figsize=(4, 4))
    plt.imshow(label)
    plt.axis("off")
    plt.colorbar()
    plt.show()
    if return_value:
        return label

def vis_image_and_label(img, label, cdl_colors=CDL, bbox_anchor=(0.5, -0.2)):
    """
    Visualize image and label side by side with legend below.
    """
    # with rasterio.open(label_path) as src:
    #     label = src.read()
    # Prepare image
    if img.shape[0] == 3 or img.shape[0] < img.shape[1]:
        img = np.transpose(img, (1, 2, 0))
    
    if img.dtype != np.uint8:
        img_normalized = np.zeros_like(img, dtype=np.float32)
        for i in range(img.shape[2]):
            channel = img[:, :, i]
            p_low = np.percentile(channel, 2)
            p_high = np.percentile(channel, 98)
            clipped = np.clip(channel, p_low, p_high)
            img_normalized[:, :, i] = (clipped - p_low) / (p_high - p_low)
        img = img_normalized
        
    img = np.clip(img, 0, 1)
    
    # Prepare label
    if label.ndim == 3 and label.shape[0] == 1:
        label = label.squeeze(0)
    
    h, w = label.shape
    rgb_label = np.zeros((h, w, 3), dtype=np.uint8)
    
    unique_classes = np.unique(label)
    legend_items = []
    
    for class_idx in unique_classes:
        if class_idx in cdl_colors:
            class_name, hex_color = cdl_colors[class_idx]
            hex_color = hex_color.lstrip('#')
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            
            mask = label == class_idx
            rgb_label[mask] = [r, g, b]
            legend_items.append((class_name, f'#{hex_color}'))
    
    # Create figure with more height for legend
    fig = plt.figure(figsize=(10, 7))
    
    # Image subplot
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(img)
    ax1.axis("off")
    
    # Label subplot
    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(rgb_label)
    ax2.axis("off")
    
    # Create legend patches
    patches = [mpatches.Patch(color=color, label=name) 
               for name, color in legend_items]
    
    # Add legend below both images with more space
    fig.legend(handles=patches, loc='lower center', 
               ncol=min(len(patches), 4), frameon=True, 
               fontsize=10, bbox_to_anchor=bbox_anchor)
    
    # Adjust layout to prevent overlap - leave more space at bottom
    plt.tight_layout(rect=[0, 0.18, 1, 1])
    plt.show()
    
def _prepare_img(img):
    if img.shape[0] == 3 or img.shape[0] < img.shape[1]:
        img = np.transpose(img, (1, 2, 0))
    
    if img.dtype != np.uint8:
        img_normalized = np.zeros_like(img, dtype=np.float32)
        for i in range(img.shape[2]):
            channel = img[:, :, i]
            p_low = np.percentile(channel, 2)
            p_high = np.percentile(channel, 98)
            clipped = np.clip(channel, p_low, p_high)
            img_normalized[:, :, i] = (clipped - p_low) / (p_high - p_low)
        img = img_normalized
    img = np.clip(img, 0, 1)
    return img

def _prepare_label(label, colors=CDL):
    # Prepare label
    if label.ndim == 3 and label.shape[0] == 1:
        label = label.squeeze(0)
    
    h, w = label.shape
    rgb_label = np.zeros((h, w, 3), dtype=np.uint8)
    
    unique_classes = np.unique(label)
    legend_items = []
    
    for class_idx in unique_classes:
        if class_idx in colors:
            class_name, hex_color = colors[class_idx]
            hex_color = hex_color.lstrip('#')
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            
            mask = label == class_idx
            rgb_label[mask] = [r, g, b]
            legend_items.append((class_name, f'#{hex_color}'))
    return rgb_label, legend_items
    
def vis_image_and_label_by_meta_lists(imgs, labels, colors=CDL, bbox_anchor=(0.5, -0.2)):
    img_years = list(imgs.keys())
    label_years = list(labels.keys())
    all_years = list(set(img_years + label_years))
    all_years.sort()
    n_cols = len(all_years)
    n_rows = 2
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4+4))
    legend_items = set()
    for i, year in enumerate(all_years):
        if year in imgs:
            img = imgs[year]
            img = _prepare_img(img)
            axs[0, i].imshow(img)
            axs[0, i].set_title(f"Input: Year {year}", fontsize=20)
        else:
            axs[0, i].imshow(np.ones((128, 128, 3)))
            axs[0, i].set_title(f"Target: Year {year}", fontsize=20)
        axs[0, i].axis("off")
        if year in labels:
            label = labels[year]
            label, _legend_items = _prepare_label(label, colors=colors)
            legend_items.update(_legend_items)
            axs[1, i].imshow(label)
        else:
            axs[1, i].imshow(np.ones((128, 128, 3)))
        axs[1, i].axis("off")
    # Create legend patches
    patches = [mpatches.Patch(color=color, label=name) 
               for name, color in legend_items]
    fig.legend(handles=patches, loc='lower center', 
               ncol=min(len(patches), 4), frameon=True, 
               fontsize=20, bbox_to_anchor=bbox_anchor)
    plt.tight_layout(rect=[0, 0.18, 1, 1])
    plt.show()
    
def vis_images_and_one_label(imgs, label, timestamps=None, label_timestamp=None, colors=CDL, max_n_cols=8, bbox_anchor=(0.5, -0.2)):
    n_imgs = imgs.shape[0]
    n_cols = n_imgs + 1
    n_rows = 1
    if n_cols > max_n_cols:
        n_rows = math.ceil(n_cols / max_n_cols)
        n_cols = max_n_cols
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 6+2))
    axs = axs.flatten()
    legend_items = set()
    assert n_imgs == len(timestamps), "The number of images and timestamps must be the same"
    for i in range(n_imgs):
        img = imgs[i]
        year = timestamps[i] // 365 + 2000
        day = timestamps[i] % 365

        img = _prepare_img(img)
        # clip to 0-1
        img = np.clip(img, 0, 1)
        axs[i].imshow(img)
        axs[i].axis("off")
        axs[i].set_title(f"Input: Year {year}, Day {day}", fontsize=20)
    for i in range(n_rows*n_cols):
        if i > n_imgs:
            axs[i].axis("off")
    label, _legend_items = _prepare_label(label, colors=colors)
    legend_items.update(_legend_items)
    axs[n_imgs].imshow(label)
    axs[n_imgs].axis("off")
    label_year = label_timestamp // 365 + 2000
    label_day = label_timestamp % 365
    axs[n_imgs].set_title(f"Label: Year {label_year}", fontsize=20)
    # Create legend patches
    patches = [mpatches.Patch(color=color, label=name) 
               for name, color in legend_items]
    fig.legend(handles=patches, loc='lower center', 
               ncol=min(len(patches), 4), frameon=True, bbox_to_anchor=bbox_anchor,
               fontsize=20)
    plt.tight_layout(rect=[0, 0.18, 1, 1])
    plt.show()
    
def vis_change_detection(imgs, labels):
    assert len(imgs) == len(labels) == 2
    (img1, year1), (img2, year2) = imgs
    label1, label2 = labels
    img1 = _prepare_img(img1)
    img2 = _prepare_img(img2)
    label1, legend_items1 = _prepare_label(label1)
    label2, legend_items2 = _prepare_label(label2)
    label_diff = np.abs(label1 - label2)  # Per-pixel absolute difference
    label_diff = label_diff.sum(axis=-1)
    # convert to binary
    label_diff = label_diff > 0
    # vis
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(img1)
    axs[0].axis("off")
    axs[0].set_title(f"Input: Year {year1}")
    axs[1].imshow(img2)
    axs[1].axis("off")
    axs[1].set_title(f"Input: Year {year2}")
    axs[2].imshow(label_diff, cmap='gray')  # Use viridis colormap for numerical differences
    axs[2].axis("off")
    axs[2].set_title("Target: Land Cover Changed")
    plt.show()

###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################

def build_hash_index(dataset, index_file='dataset_hash_index.pkl'):
    """
    Build hash-based index for O(1) lookups.
    """
    print("Building hash index...")
    
    hash_map = {}
    
    for i in tqdm(range(len(dataset)), desc="Indexing dataset"):
        sample = dataset[i]
        geo = sample['geo']
        
        # Create unique hash from bounds (rounded to avoid floating point issues)
        bounds = tuple(round(b, 3) for b in geo['bounds'])
        crs = geo['crs_epsg']
        
        # Use (crs, bounds) as key
        key = (crs, bounds)
        hash_map[key] = i
    
    # Save index
    with open(index_file, 'wb') as f:
        pickle.dump(hash_map, f)
    
    print(f"✓ Hash index with {len(hash_map)} entries saved")
    return hash_map


def find_matching_image_instant(label_path, dataset, hash_map, decimals=3):
    """
    Instant O(1) lookup using hash map.
    """
    with open(hash_map, "rb") as f:
        hash_map = pickle.load(f)
    with rasterio.open(label_path) as src:
        label = src.read()
        target_crs = src.crs.to_epsg()
        target_bounds = tuple(round(b, decimals) for b in src.bounds)
    
    key = (target_crs, target_bounds)
    
    if key in hash_map:
        idx = hash_map[key]
        print(f"✓ Found match at index {idx} (instant lookup)")
        sample = dataset[idx]
        img = sample['optical']
        return idx, img, label, sample['geo']
    
    print("✗ No match found")
    return None, None, None, None

def find_image_by_meta(meta, label_path_template, dataset, hash_map, decimals=3, save_days=False):
    with open(hash_map, "rb") as f:
        hash_map = pickle.load(f)
    imgs = {}
    for _, meta_row in meta.iterrows():
        crs = meta_row['crs_epsg']
        bounds = tuple(round(b, decimals) for b in meta_row['bounds'])
        key = (crs, bounds)
        if key in hash_map:
            idx = hash_map[key]
            year = meta_row['year']
            day = meta_row['day']
            sample = dataset[idx]
            img = sample['optical']
            if year not in imgs:
                imgs[year] = []
            if save_days:
                imgs[year].append((img, day))
            else:
                imgs[year].append(img)
            # imgs[year].append(img)
    loc_uid = meta['location_uid'][0]
    label_paths = label_path_template.format(loc_uid=loc_uid)
    label_paths = sorted(glob.glob(label_paths))
    labels = {}
    for label_path in label_paths:
        with rasterio.open(label_path) as src:
            label = src.read()
            year = int(label_path.split('/')[-1].split('_')[1].split('.')[0])
            labels[year] = label
    return imgs, labels
    # return None, None
    
def visualize_static_dataset(data, colors=CDL, bbox_anchor=(0.5, -0.2)):
    opticals = data['optical']
    label = data['label']
    vis_image_and_label(opticals, label, colors, bbox_anchor)

def visualize_temporal_dataset(data, colors=CDL, bbox_anchor=(0.5, -0.2)):
    opticals = data['optical']
    label = data['label']
    timestamps = data['timestamp']
    label_timestamp = data['label_timestamp']
    vis_images_and_one_label(opticals, label, timestamps, label_timestamp, colors, bbox_anchor=bbox_anchor)
    
def visualize_multilabel_static_dataset(data, label_mapping, bbox_anchor=(0.5, -0.2)):
    label = data['label']
    label_timestamp = data['label_timestamp']
    optical = data['optical']
    # convert one-hot multi-label to class name
    label_text = []
    n_used_labels = 0
    for i in range(label.shape[0]):
        if label[i] == 1:
            n_used_labels += 1
            if n_used_labels % 2 == 0:
                label_text.append(label_mapping[i]+',\n')
            else:
                label_text.append(label_mapping[i]+',')
    # join the label text and drop the last comma
    label_text = ' '.join(label_text)[:-1]
    caption = f"Year: {label_timestamp // 365 + 2000}\n Labels: {label_text}"
    vis_image(optical, caption=caption, bbox_anchor=bbox_anchor)
    
def visualize_multilabel_temporal_dataset(data, label_mapping, bbox_anchor=(0.5, -0.2)):
    label = data['label']
    label_timestamp = data['label_timestamp']
    timestamps = data['timestamp']
    opticals = data['optical']
    # convert one-hot multi-label to class name
    label_text = []
    n_used_labels = 0
    for i in range(label.shape[0]):
        if label[i] == 1:
            n_used_labels += 1
            if n_used_labels % 5 == 0:
                label_text.append(label_mapping[i]+',\n')
            else:
                label_text.append(label_mapping[i]+',')
    # join the label text and drop the last comma
    label_text = ' '.join(label_text)[:-1]
    caption = f"Label Year: {label_timestamp // 365 + 2000}\n Labels: {label_text}"
    vis_images(opticals, timestamps=timestamps, captions=caption, bbox_anchor=bbox_anchor)
    
def visualize_static_regression_dataset(data, bbox_anchor=(0.5, 0.2)):
    label = data['label']
    opticals = data['optical'].transpose(1, 2, 0)
    img = _prepare_img(opticals)
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(img)
    axs[0].axis('off')
    axs[0].set_title('Input', fontsize=20)
    axs[1].imshow(label)
    axs[1].axis('off')
    axs[1].set_title('Label', fontsize=20)
    plt.show()
    
def visualize_temporal_regression_dataset(data, bbox_anchor=(0.5, 0.2)):
    label = data['label']
    opticals = data['optical']
    timestamps = data['timestamp']
    label_timestamp = data['label_timestamp']
    n_imgs = len(opticals)
    n_cols = n_imgs + 1
    fig, axs = plt.subplots(1, n_cols, figsize=(n_cols * 4, 6))
    for i in range(n_imgs):
        img = opticals[i]
        img = _prepare_img(img)
        axs[i].imshow(img)
        axs[i].axis('off')
        axs[i].set_title(f'Year {timestamps[i] // 365 + 2000}, Day {timestamps[i] % 365}', fontsize=20)
    axs[n_imgs].imshow(label)
    axs[n_imgs].axis('off')
    axs[n_imgs].set_title(f'Label: Year {label_timestamp // 365 + 2000}', fontsize=20)
    plt.show()