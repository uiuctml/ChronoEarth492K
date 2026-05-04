import torch
import numpy as np


def temporal_mae_collate_fn(batch):
    """
    Collate function for TemporalChronoEarth batches.

    Each example has:
      optical:     (T, C, H, W) numpy array
      timestamp:   (T,) array of day-of-year values
      num_frames:  int T
      optical_channel_wv: list of wavelengths

    Returns a dict with:
      optical:      (B, T_max, C, H, W)
      timestamps:   (B, T_max, 2)  — [year, doy] per frame, padded with -1
      valid_mask:   (B, T_max) bool — True = real frame
    """
    Ts = [ex["num_frames"] for ex in batch]
    T_max = max(Ts)
    B = len(batch)

    # Build tensors from first valid example for shape info
    first = batch[0]
    C, H, W = first["optical"].shape[1], first["optical"].shape[2], first["optical"].shape[3]

    optical_out = torch.zeros(B, T_max, C, H, W, dtype=torch.float32)
    timestamps_out = torch.full((B, T_max, 2), fill_value=-1, dtype=torch.long)
    valid_mask = torch.zeros(B, T_max, dtype=torch.bool)

    for b, ex in enumerate(batch):
        T = ex["num_frames"]
        optical_out[b, :T] = torch.as_tensor(ex["optical"], dtype=torch.float32)

        # timestamps from TemporalChronoEarth are day-of-year integers
        # We convert to (year, doy) pairs using the metadata year/day columns.
        # If only a single integer timestamp is provided, treat as absolute day index.
        ts = ex["timestamp"]                      # (T,) numpy or list
        ts = np.array(ts, dtype=np.int64)

        # Timestamps are YYYYDDD integers (e.g. 2008337 = year 2008, doy 337)
        year = ts // 1000
        doy = ts % 1000
        timestamps_out[b, :T, 0] = torch.as_tensor(year, dtype=torch.long)
        timestamps_out[b, :T, 1] = torch.as_tensor(doy, dtype=torch.long)

        valid_mask[b, :T] = True

    return {
        "optical": optical_out,
        "timestamps": timestamps_out,
        "valid_mask": valid_mask,
    }
