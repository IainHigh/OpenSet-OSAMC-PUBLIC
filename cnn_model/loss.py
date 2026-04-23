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


def _superclass_ids_from_multihot(labels: Tensor) -> Tensor:
    """Build per-sample superclass IDs from multi-hot labels.
    Samples with identical active class index sets receive the same superclass ID.
    """

    label_bool = labels > 0.0
    keys = [
        tuple(torch.nonzero(row, as_tuple=False).view(-1).tolist())
        for row in label_bool
    ]

    key_to_id: Dict[Tuple[int, ...], int] = {}
    ids = []
    next_id = 0
    for key in keys:
        if key not in key_to_id:
            key_to_id[key] = next_id
            next_id += 1
        ids.append(key_to_id[key])

    return torch.tensor(ids, dtype=torch.long, device=labels.device)


def batch_hard_triplet_loss(
    embeddings: Tensor,
    labels: Tensor,
    superclass_ids: Tensor,
    margin: float = 1.0,
    normalize_embeddings: bool = True,
    hard_positive_quantile: float = 0.9,
    hard_negative_quantile: float = 0.1,
    p: float = 2.0,
) -> Tensor:
    """
    Compute a batch-hard triplet loss for multi-label targets.

    For each anchor, selects a hard positive (high-distance quantile among
    identical-label samples) and a hard negative (low-distance quantile among
    disjoint-label samples) to form the triplet.

    Args:
        embeddings: Tensor of shape (batch, embedding_dim).
        labels: Float tensor of shape (batch, num_classes) with multi-hot labels.
        superclass_ids: Long tensor of shape (batch,) with superclass IDs
            derived from multi-hot labels.
        margin: Margin for triplet separation.
        normalize_embeddings: If True, L2-normalize embeddings before distance.
        hard_positive_quantile: Quantile in [0, 1] used to pick hard positives.
            1.0 recovers classic batch-hard max positive.
        hard_negative_quantile: Quantile in [0, 1] used to pick hard negatives.
            0.0 recovers classic batch-hard min negative.
        p: Norm degree for pairwise distance (passed to torch.cdist).

    Returns:
        Scalar tensor representing the triplet loss.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"Expected embeddings to be 2D, got {embeddings.ndim}D.")
    if labels.ndim != 2 or labels.shape[0] != embeddings.shape[0]:
        raise ValueError("Labels must be 2D with the same batch size as embeddings.")
    if not 0.0 <= hard_positive_quantile <= 1.0:
        raise ValueError("hard_positive_quantile must be in [0, 1].")
    if not 0.0 <= hard_negative_quantile <= 1.0:
        raise ValueError("hard_negative_quantile must be in [0, 1].")

    if normalize_embeddings:
        embeddings = F.normalize(embeddings, p=2, dim=1)

    device = embeddings.device
    batch_size = embeddings.size(0)
    distances = torch.cdist(embeddings, embeddings, p=p)

    if superclass_ids is None:
        superclass_ids = _superclass_ids_from_multihot(labels)
    if superclass_ids.ndim != 1 or superclass_ids.shape[0] != batch_size:
        raise ValueError(
            "superclass_ids must be 1D with the same batch size as embeddings."
        )
    positive_mask = superclass_ids.unsqueeze(1) == superclass_ids.unsqueeze(0)

    diag = torch.eye(batch_size, dtype=torch.bool, device=device)
    positive_mask = positive_mask & ~diag

    # Negatives are only samples with completely disjoint active-label sets.
    # This avoids conflicting supervision for partially overlapping superclasses.
    label_bool = labels > 0.0
    shared_label_mask = (label_bool.float() @ label_bool.float().transpose(0, 1)) > 0
    negative_mask = (~shared_label_mask) & ~diag

    anchor_losses = []
    for anchor in range(batch_size):
        pos_vals = distances[anchor][positive_mask[anchor]]
        neg_vals = distances[anchor][negative_mask[anchor]]
        if pos_vals.numel() == 0 or neg_vals.numel() == 0:
            continue

        hard_pos = torch.quantile(pos_vals, hard_positive_quantile)
        hard_neg = torch.quantile(neg_vals, hard_negative_quantile)
        anchor_losses.append(torch.relu(hard_pos - hard_neg + margin))

    if not anchor_losses:
        return embeddings.new_tensor(0.0)

    return torch.stack(anchor_losses).mean()


def combined_bce_triplet_loss(
    logits: Tensor,
    labels: Tensor,
    embeddings: Tensor,
    lambda_bce: float = 1.0,
    lambda_tri: float = 1.0,
    margin: float = 1.0,
    pos_weight: Optional[Tensor] = None,
    normalize_embeddings: bool = True,
    hard_positive_quantile: float = 0.9,
    hard_negative_quantile: float = 0.1,
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
        hard_positive_quantile: Quantile used to select hard positives.
        hard_negative_quantile: Quantile used to select hard negatives.

    Returns:
        Tuple of (total_loss, component_losses).
        component_losses contains the unweighted BCE and triplet losses.
    """
    bce_loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)

    if lambda_tri <= 0.0:
        triplet_loss = embeddings.new_tensor(0.0)
    else:
        superclass_ids = _superclass_ids_from_multihot(labels)
        triplet_loss = batch_hard_triplet_loss(
            embeddings,
            labels,
            superclass_ids=superclass_ids,
            margin=margin,
            normalize_embeddings=normalize_embeddings,
            hard_positive_quantile=hard_positive_quantile,
            hard_negative_quantile=hard_negative_quantile,
        )
    total_loss = lambda_bce * bce_loss + lambda_tri * triplet_loss
    return total_loss, {"bce": bce_loss, "triplet": triplet_loss}
