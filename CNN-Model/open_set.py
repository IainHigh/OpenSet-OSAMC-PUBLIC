"""
open_set.py:
Feature-driven open-set recognition utilities for the modulation classifier.

The previous implementation relied on simple logit statistics (e.g. energy,
range) to gate predictions, which proved fragile when the composition of known
and unknown modulations changed. This module replaces that heuristic with a
feature-space model that better captures the multi-label structure of the
problem.

For each class we estimate a diagonal Gaussian model over the penultimate
feature vectors of training samples where the class is present. Quantile-based
Mahalanobis distance thresholds are then used to (a) reject low-likelihood
class predictions and (b) flag samples whose closest class prototype is still
unlikely. This aligns with modern open-set recognition practice, where
distance-aware feature modeling often outperforms raw logit thresholding,
especially in multi-label regimes.
"""

# pylint: disable=import-error

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class _ClassFeatureModel:
    """Diagonal Gaussian model for a single class's feature distribution."""

    mean: torch.Tensor
    var: torch.Tensor
    threshold: float
    eps: float = 1e-6

    def mahalanobis(self, features: torch.Tensor) -> torch.Tensor:
        """Return squared Mahalanobis distance(s) to the class centroid."""

        diff = features - self.mean
        inv_var = 1.0 / (self.var + self.eps)
        return (diff * diff * inv_var).sum(dim=-1)


class FeatureBasedOpenSetDetector:
    """Open-set detector that operates in the model's penultimate feature space."""

    def __init__(
        self, quantile: float = 0.99, min_samples: int = 5, eps: float = 1e-6
    ) -> None:
        if not 0.5 < quantile < 1.0:
            raise ValueError(
                "Quantile must lie between 0.5 and 1.0 for upper-tail thresholds."
            )
        if min_samples < 1:
            raise ValueError("min_samples must be at least 1.")
        self.quantile = quantile
        self.min_samples = min_samples
        self.eps = eps
        self._class_models: Dict[int, _ClassFeatureModel] = {}

    def fit(
        self, model: nn.Module, data_loader: DataLoader, device: torch.device
    ) -> None:
        """Calibrate class-conditional thresholds in feature space."""

        if not hasattr(model, "forward_features"):
            raise AttributeError(
                "Model must expose a `forward_features` method for OSR calibration."
            )

        was_training = model.training
        model.eval()

        num_classes = getattr(data_loader.dataset, "num_classes", None)
        if num_classes is None:
            raise AttributeError("DataLoader dataset must define `num_classes`.")

        class_sums: Optional[torch.Tensor] = None
        class_sumsq: Optional[torch.Tensor] = None
        class_counts = torch.zeros(num_classes, dtype=torch.long)

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Calibrating OSR detector"):
                inputs, labels = batch[0].to(device, non_blocking=True), batch[1]
                labels = labels.to(device, non_blocking=True)
                _, features = model.forward_features(inputs)
                feats_cpu = features.detach().to("cpu")
                labels_cpu = labels.detach().to("cpu")

                if class_sums is None:
                    feature_dim = feats_cpu.shape[1]
                    class_sums = torch.zeros(
                        (num_classes, feature_dim), dtype=torch.float64
                    )
                    class_sumsq = torch.zeros_like(class_sums)

                positive_mask = labels_cpu > 0.5
                for class_idx in range(num_classes):
                    mask = positive_mask[:, class_idx]
                    if mask.any():
                        selected = feats_cpu[mask].to(dtype=torch.float64)
                        class_counts[class_idx] += selected.shape[0]
                        class_sums[class_idx] += selected.sum(dim=0)
                        class_sumsq[class_idx] += (selected * selected).sum(dim=0)

        if class_sums is None or class_sumsq is None:
            raise RuntimeError(
                "No feature statistics were accumulated during OSR calibration."
            )

        for class_idx in range(num_classes):
            if class_counts[class_idx] < self.min_samples:
                continue
            mean = class_sums[class_idx] / class_counts[class_idx]
            var = class_sumsq[class_idx] / class_counts[class_idx] - mean * mean
            var = torch.clamp(var, min=self.eps)
            self._class_models[class_idx] = _ClassFeatureModel(
                mean=mean, var=var, threshold=0.0, eps=self.eps
            )

        if not self._class_models:
            raise RuntimeError(
                "Insufficient per-class samples to fit any OSR feature models."
            )

        class_distances: Dict[int, List[torch.Tensor]] = {
            idx: [] for idx in self._class_models
        }
        sample_best_distances: List[torch.Tensor] = []

        with torch.no_grad():
            for batch in data_loader:
                inputs, labels = batch[0].to(device, non_blocking=True), batch[1]
                labels = labels.to(device, non_blocking=True)
                _, features = model.forward_features(inputs)
                feats_cpu = features.detach().to("cpu")
                labels_cpu = labels.detach().to("cpu")

                for class_idx, model_stats in self._class_models.items():
                    mask = labels_cpu[:, class_idx] > 0.5
                    if mask.any():
                        dists = model_stats.mahalanobis(feats_cpu[mask])
                        class_distances[class_idx].append(dists)

                positive_mask = labels_cpu > 0.5
                for sample_idx, row_mask in enumerate(positive_mask):
                    if not row_mask.any():
                        continue
                    per_class_dists = []
                    for class_idx in torch.nonzero(row_mask, as_tuple=False).view(-1):
                        model_stats = self._class_models.get(int(class_idx))
                        if model_stats is None:
                            continue
                        dist_val = model_stats.mahalanobis(
                            feats_cpu[sample_idx : sample_idx + 1]
                        )
                        per_class_dists.append(dist_val[0])
                    if per_class_dists:
                        sample_best_distances.append(torch.stack(per_class_dists).min())

        for class_idx, dist_list in class_distances.items():
            if not dist_list:
                continue
            concatenated = torch.cat(dist_list)
            threshold = torch.quantile(concatenated, self.quantile).item()
            self._class_models[class_idx].threshold = float(threshold)

        if was_training:
            model.train()

    def filter_predictions(
        self, features: torch.Tensor, decisions: torch.Tensor
    ) -> torch.Tensor:
        """Suppress class predictions that are implausible in feature space."""

        if not self._class_models:
            raise RuntimeError("OSR detector has not been fitted yet.")

        filtered = decisions.clone()
        feat_cpu = features.detach().to("cpu")
        for class_idx, model_stats in self._class_models.items():
            if filtered[class_idx] <= 0.0:
                continue
            dist_val = model_stats.mahalanobis(feat_cpu.unsqueeze(0))[0].item()
            if dist_val > model_stats.threshold:
                filtered[class_idx] = 0.0

        return filtered

    def get_class_statistics(self) -> Dict[int, Dict[str, float]]:
        """Return calibrated thresholds for each class."""

        if not self._class_models:
            raise RuntimeError("OSR detector has not been fitted yet.")
        return {
            idx: {"threshold": model_stats.threshold}
            for idx, model_stats in self._class_models.items()
        }
