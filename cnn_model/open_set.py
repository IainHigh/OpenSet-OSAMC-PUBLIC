"""
open_set.py:
Feature-driven open-set recognition utilities for the modulation classifier.

For each modulation superclass (e.g., BPSK+QPSK) we estimate a diagonal Gaussian
model over penultimate feature vectors from training samples in that superclass.
Quantile-based Mahalanobis thresholds are then used to reject samples that are
unlikely to belong to any known superclass.
"""

# pylint: disable=import-error

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class _SuperclassFeatureModel:
    """Diagonal Gaussian model for a single superclass feature distribution."""

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
        self._superclass_models: Dict[Tuple[int, ...], _SuperclassFeatureModel] = {}

    @staticmethod
    def _superclass_key_from_label_row(label_row: torch.Tensor) -> Tuple[int, ...]:
        """Map a multi-hot known-label row to a deterministic superclass key."""

        return tuple(torch.nonzero(label_row > 0.5, as_tuple=False).view(-1).tolist())

    def fit(
        self, model: nn.Module, data_loader: DataLoader, device: torch.device
    ) -> None:
        """Calibrate superclass-conditional thresholds in feature space."""

        if not hasattr(model, "forward_features"):
            raise AttributeError(
                "Model must expose a `forward_features` method for OSR calibration."
            )

        was_training = model.training
        model.eval()
        superclass_sums: Dict[Tuple[int, ...], torch.Tensor] = {}
        superclass_sumsq: Dict[Tuple[int, ...], torch.Tensor] = {}
        superclass_counts: Dict[Tuple[int, ...], int] = {}
        feature_dim: Optional[int] = None

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Calibrating OSR detector"):
                inputs, labels = batch[0].to(device, non_blocking=True), batch[1]
                labels = labels.to(device, non_blocking=True)
                _, features = model.forward_features(inputs)
                feats_cpu = features.detach().to("cpu")
                labels_cpu = labels.detach().to("cpu")

                if feature_dim is None:
                    feature_dim = feats_cpu.shape[1]

                for sample_idx in range(labels_cpu.shape[0]):
                    key = self._superclass_key_from_label_row(labels_cpu[sample_idx])
                    if not key:
                        continue

                    feat = feats_cpu[sample_idx].to(dtype=torch.float64)
                    if key not in superclass_sums:
                        assert feature_dim is not None
                        superclass_sums[key] = torch.zeros(
                            feature_dim, dtype=torch.float64
                        )
                        superclass_sumsq[key] = torch.zeros(
                            feature_dim, dtype=torch.float64
                        )
                        superclass_counts[key] = 0

                    superclass_counts[key] += 1
                    superclass_sums[key] += feat
                    superclass_sumsq[key] += feat * feat

        if not superclass_sums:
            raise RuntimeError(
                "No feature statistics were accumulated during OSR calibration."
            )

        for key, count in superclass_counts.items():
            if count < self.min_samples:
                continue
            mean = superclass_sums[key] / count
            var = superclass_sumsq[key] / count - mean * mean
            var = torch.clamp(var, min=self.eps)
            self._superclass_models[key] = _SuperclassFeatureModel(
                mean=mean, var=var, threshold=0.0, eps=self.eps
            )

        if not self._superclass_models:
            raise RuntimeError(
                "Insufficient per-superclass samples to fit any OSR feature models."
            )

        superclass_distances: Dict[Tuple[int, ...], List[torch.Tensor]] = {
            key: [] for key in self._superclass_models
        }

        with torch.no_grad():
            for batch in data_loader:
                inputs, labels = batch[0].to(device, non_blocking=True), batch[1]
                labels = labels.to(device, non_blocking=True)
                _, features = model.forward_features(inputs)
                feats_cpu = features.detach().to("cpu")
                labels_cpu = labels.detach().to("cpu")

                for sample_idx in range(labels_cpu.shape[0]):
                    key = self._superclass_key_from_label_row(labels_cpu[sample_idx])
                    if not key:
                        continue
                    model_stats = self._superclass_models.get(key)
                    if model_stats is None:
                        continue
                    dists = model_stats.mahalanobis(
                        feats_cpu[sample_idx : sample_idx + 1]
                    )
                    superclass_distances[key].append(dists)

        for key, dist_list in superclass_distances.items():
            if not dist_list:
                continue
            concatenated = torch.cat(dist_list)
            threshold = torch.quantile(concatenated, self.quantile).item()
            self._superclass_models[key].threshold = float(threshold)

        if was_training:
            model.train()

    def filter_predictions(
        self, features: torch.Tensor, decisions: torch.Tensor
    ) -> torch.Tensor:
        """Accept the predicted superclass only if it is in-threshold in feature space."""

        if not self._superclass_models:
            raise RuntimeError("OSR detector has not been fitted yet.")

        filtered = torch.zeros_like(decisions)
        feat_cpu = features.detach().to("cpu")
        decisions_cpu = decisions.detach().to("cpu")

        # Build the predicted superclass key from the predicted multi-hot vector.
        predicted_key = self._superclass_key_from_label_row(decisions_cpu)

        # If nothing was predicted, reject.
        if not predicted_key:
            return filtered

        # If this predicted superclass was never calibrated, reject.
        model_stats = self._superclass_models.get(predicted_key)
        if model_stats is None:
            return filtered

        dist_val = model_stats.mahalanobis(feat_cpu.unsqueeze(0))[0].item()

        # Reject if the embedding is too far from the predicted superclass cluster.
        if dist_val > model_stats.threshold:
            return filtered

        # Otherwise keep the original predicted superclass.
        return decisions.clone()

    def get_class_statistics(self) -> Dict[Tuple[int, ...], Dict[str, float]]:
        """Return calibrated thresholds for each superclass."""

        if not self._superclass_models:
            raise RuntimeError("OSR detector has not been fitted yet.")
        return {
            key: {"threshold": model_stats.threshold}
            for key, model_stats in self._superclass_models.items()
        }
