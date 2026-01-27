"""
dataset.py:
Defines a dataset for loading modulation samples with parallel metadata
extraction and background prefetching of I/Q data.
"""

# pylint: disable=import-error
import os
import json
import numpy as np
import torch
from torch.utils.data import IterableDataset


def _normalize_iq_sample(x: np.ndarray) -> np.ndarray:
    """
    Normalize an I/Q sample to zero mean and unit power.
    """

    # Ensure float32 for stable numerics and in-place operations.
    x = x.astype(np.float32, copy=False)

    # Remove any DC offset on each channel individually.
    x -= x.mean(axis=1, keepdims=True)

    # Compute RMS power per channel and normalise.
    x /= np.sqrt(np.mean(x**2, axis=1, keepdims=True)) + 1e-8

    # Guard against numerical issues that can appear in extreme cases.
    # np.nan_to_num(x, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
    return x


class ModulationDataset(IterableDataset):
    """Dataset for loading modulation data.

    The dataset parses all metadata files in parallel during
    initialization. Actual sample tensors are loaded on a background
    thread and prefetched into a queue while the consumer iterates over
    them.
    """

    def __init__(
        self,
        data_dir,
        transform=None,
        shuffle=False,
        label_to_idx=None,
    ):
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir) if f.endswith(".sigmf-data")]
        self.transform = transform
        self.shuffle = shuffle

        # metadata extraction
        meta = list(map(self._parse_meta, self.files))

        label_lists, snr_values, overlap_counts = zip(*meta) if meta else ([], [], [])
        self.snr_values = list(snr_values)
        self.overlap_counts = list(overlap_counts)

        if label_to_idx is None:
            all_labels = {label for mods in label_lists for label in mods}
            self.label_to_idx = {lab: i for i, lab in enumerate(sorted(all_labels))}
        else:
            self.label_to_idx = dict(label_to_idx)

        self.num_classes = len(self.label_to_idx)
        self.labels = []
        self.sample_unknown_mods = []

        for mods in label_lists:
            arr = np.zeros(self.num_classes, dtype=np.float32)
            unknowns = []
            for m in mods:
                idx = self.label_to_idx.get(m)
                if idx is not None:
                    arr[idx] = 1.0
                else:
                    unknowns.append(m)
            self.labels.append(arr)
            self.sample_unknown_mods.append(unknowns)

        labels_array = np.array(self.labels)
        if labels_array.size == 0:
            self.pos_weight = torch.tensor(1.0)
        else:
            pos_counts = labels_array.sum(axis=0)
            neg_counts = labels_array.shape[0] - pos_counts
            self.pos_weight = torch.tensor(
                neg_counts / (pos_counts + 1e-5), dtype=torch.float32
            )

    def _parse_meta(self, file):
        meta_path = os.path.join(
            self.data_dir, file.replace(".sigmf-data", ".sigmf-meta")
        )
        with open(meta_path, "r", encoding="utf-8") as fh:
            metadata = json.load(fh)
        mods = metadata["annotations"][0]["rfml_labels"]["modulations"]
        snr = metadata["annotations"][0]["channel"]["snr"]
        overlap = max(len(mods) - 1, 0)
        return mods, snr, overlap

    def __getitem__(self, idx):
        data_path = os.path.join(self.data_dir, self.files[idx])

        iq_data = np.load(data_path, mmap_mode="r")
        iq_data = iq_data.astype(np.float32, copy=False)

        i = iq_data[0::2]
        q = iq_data[1::2]
        x = np.stack([i, q], axis=0)  # (2, N)
        x = _normalize_iq_sample(x)

        if self.transform:
            x = self.transform(x)

        return (
            torch.from_numpy(x),  # zero-copy
            torch.from_numpy(
                self.labels[idx]
            ).float(),  # pre-store labels as np.float32 arrays in __init__
            torch.tensor(self.snr_values[idx], dtype=torch.float32),
            torch.tensor(self.overlap_counts[idx], dtype=torch.int32),
        )

    def __len__(self):
        return len(self.files)

    def __iter__(self):
        indices = list(range(len(self)))
        if self.shuffle:
            np.random.shuffle(indices)
        for idx in indices:
            yield self[idx]
