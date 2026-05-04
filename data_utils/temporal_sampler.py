import math
import random
from torch.utils.data import Sampler

class BucketBatchSampler(Sampler):
    def __init__(self, lengths, batch_size, boundaries, shuffle=True, drop_last=False):
        # boundaries: e.g. [1,2,3,4,6,8,12,16] (edges for buckets)
        self.lengths = lengths
        self.batch_size = batch_size
        self.boundaries = boundaries
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.buckets = [[] for _ in range(len(boundaries)+1)]
        for i, L in enumerate(lengths):
            b = 0
            while b < len(boundaries) and L > boundaries[b]:
                b += 1
            self.buckets[b].append(i)

    def __iter__(self):
        buckets = [b[:] for b in self.buckets]
        if self.shuffle:
            for b in buckets: random.shuffle(b)

        for b in buckets:
            for k in range(0, len(b), self.batch_size):
                batch = b[k:k+self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch

    def __len__(self):
        n = 0
        for b in self.buckets:
            n += len(b) // self.batch_size if self.drop_last else math.ceil(len(b)/self.batch_size)
        return n
    
import math
import random
from torch.utils.data import Sampler

class AdaptiveBucketBatchSampler(Sampler):
    def __init__(
        self,
        lengths,
        max_frames,
        boundaries,
        shuffle=True,
        shuffle_buckets=True,
        drop_last=False,
        min_bs=1,
        max_bs=None,
        seed=0,
        last_bucket_rep=None,  # optional: override representative for last bucket
    ):
        self.lengths = lengths
        self.max_frames = int(max_frames)
        self.boundaries = list(boundaries)
        self.shuffle = shuffle
        self.shuffle_buckets = shuffle_buckets
        self.drop_last = drop_last
        self.min_bs = int(min_bs)
        self.max_bs = None if max_bs is None else int(max_bs)
        self.seed = seed
        self.epoch = 0

        # Build buckets
        self.buckets = [[] for _ in range(len(self.boundaries) + 1)]
        for i, T in enumerate(self.lengths):
            b = 0
            while b < len(self.boundaries) and T > self.boundaries[b]:
                b += 1
            self.buckets[b].append(i)

        # Representative size per bucket: upper bound edge; last bucket uses max observed (or user override)
        max_T = max(self.lengths)
        if last_bucket_rep is None:
            last_bucket_rep = max_T  # or you can set to self.boundaries[-1] to cap more aggressively

        self.bucket_rep = []
        for b in range(len(self.boundaries) + 1):
            if b < len(self.boundaries):
                rep = self.boundaries[b]
            else:
                rep = last_bucket_rep
            self.bucket_rep.append(max(1, int(rep)))

        # Precompute batch size per bucket, with clamping and safety
        self.bucket_bs = []
        for rep in self.bucket_rep:
            bs = self.max_frames // rep
            bs = max(self.min_bs, bs)
            if self.max_bs is not None:
                bs = min(self.max_bs, bs)
            # Safety: ensure bs >= 1 always
            bs = max(1, bs)
            self.bucket_bs.append(bs)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)

        buckets = [b[:] for b in self.buckets]
        if self.shuffle:
            for b in buckets:
                rng.shuffle(b)

        bucket_order = list(range(len(buckets)))
        if self.shuffle_buckets and self.shuffle:
            rng.shuffle(bucket_order)

        for bi in bucket_order:
            idxs = buckets[bi]
            bs = self.bucket_bs[bi]
            for k in range(0, len(idxs), bs):
                batch = idxs[k:k + bs]
                if len(batch) < bs and self.drop_last:
                    continue
                yield batch

    def __len__(self):
        n = 0
        for bi, b in enumerate(self.buckets):
            bs = self.bucket_bs[bi]
            n += (len(b) // bs) if self.drop_last else math.ceil(len(b) / bs)
        return n