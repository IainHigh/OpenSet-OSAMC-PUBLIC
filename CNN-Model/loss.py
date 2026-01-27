"""
loss.py:
Defines loss functions for the CNN modulation classifier, including a
combined binary cross entropy and triplet loss.
"""

# pylint: disable=import-error
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def _pairwise_distance(embeddings: Tensor, p: float = 2.0) -> Tensor:
    """Compute pairwise distances between embeddings."""
    return torch.cdist(embeddings, embeddings, p=p)


def batch_hard_triplet_loss(
    embeddings: Tensor,
    labels: Tensor,
    margin: float = 1.0,
    normalize_embeddings: bool = True,
    p: float = 2.0,
) -> Tensor:
    """
    Compute a batch-hard triplet loss for multi-label targets.

    For each anchor, selects the hardest positive (furthest with identical
    label set) and hardest negative (closest with no shared labels) to form the triplet.

    Args:
        embeddings: Tensor of shape (batch, embedding_dim).
        labels: Float tensor of shape (batch, num_classes) with multi-hot labels.
        margin: Margin for triplet separation.
        normalize_embeddings: If True, L2-normalize embeddings before distance.
        p: Norm degree for pairwise distance (passed to torch.cdist).

    Returns:
        Scalar tensor representing the triplet loss.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"Expected embeddings to be 2D, got {embeddings.ndim}D.")
    if labels.ndim != 2 or labels.shape[0] != embeddings.shape[0]:
        raise ValueError("Labels must be 2D with the same batch size as embeddings.")

    if normalize_embeddings:
        embeddings = F.normalize(embeddings, p=2, dim=1)

    device = embeddings.device
    batch_size = embeddings.size(0)
    distances = _pairwise_distance(embeddings, p=p)

    label_bool = labels > 0.0
    positive_mask = (label_bool.unsqueeze(1) == label_bool.unsqueeze(0)).all(dim=2)
    diag = torch.eye(batch_size, dtype=torch.bool, device=device)
    positive_mask = positive_mask & ~diag
    negative_mask = ~positive_mask & ~diag

    # Hardest positive: maximum distance among positives
    pos_dist = distances.masked_fill(~positive_mask, float("-inf"))
    hardest_pos = pos_dist.max(dim=1).values

    # Hardest negative: minimum distance among negatives
    neg_dist = distances.masked_fill(~negative_mask, float("inf"))
    hardest_neg = neg_dist.min(dim=1).values

    valid = torch.isfinite(hardest_pos) & torch.isfinite(hardest_neg)
    if not valid.any():
        return embeddings.new_tensor(0.0)

    loss = torch.relu(hardest_pos[valid] - hardest_neg[valid] + margin)
    return loss.mean()


def combined_bce_triplet_loss(
    logits: Tensor,
    labels: Tensor,
    embeddings: Tensor,
    lambda_bce: float = 1.0,
    lambda_tri: float = 1.0,
    margin: float = 1.0,
    pos_weight: Optional[Tensor] = None,
    normalize_embeddings: bool = True,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """
    Compute a combined loss of BCEWithLogits and triplet loss.

    Args:
        logits: Model logits of shape (batch, num_classes).
        labels: Ground truth multi-hot labels of shape (batch, num_classes).
        embeddings: Feature embeddings of shape (batch, embedding_dim).
        lambda_bce: Weight for BCE loss component.
        lambda_tri: Weight for triplet loss component.
        margin: Margin for triplet loss.
        pos_weight: Optional class weighting tensor for BCE.
        normalize_embeddings: If True, L2-normalize embeddings before distance.

    Returns:
        Tuple of (total_loss, component_losses).
        component_losses contains the unweighted BCE and triplet losses.
    """
    bce_loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
    triplet_loss = batch_hard_triplet_loss(
        embeddings, labels, margin=margin, normalize_embeddings=normalize_embeddings
    )
    total_loss = lambda_bce * bce_loss + lambda_tri * triplet_loss
    return total_loss, {"bce": bce_loss, "triplet": triplet_loss}
