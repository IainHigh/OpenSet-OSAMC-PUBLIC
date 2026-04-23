"""
main.py:
Entry point for CNN-Model training.
"""

# pylint: disable=import-error

# Standard Imports
import argparse
import math
from pathlib import Path
import uuid
from typing import Dict, List, Optional, Tuple, Sequence

# Third Party Imports
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch
from torch.utils.data import DataLoader
from torch import optim, nn
from tqdm import tqdm
import numpy as np
from previous_sota_results import PREVIOUS_SOTA_ACCURACY, PREVIOUS_SOTA_SNRS

# Local Imports
from .dataset import ModulationDataset
from .classifier import ModulationClassifier
from .open_set import FeatureBasedOpenSetDetector
from .loss import combined_bce_triplet_loss
from .configs import Config

c = Config()
matplotlib.use("Agg")


def set_rng_seed(seed: int):
    """Set the random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def _train_model(train_loader, model):
    optimizer = optim.Adam(model.parameters(), lr=c.LEARNING_RATE)
    model.to(c.DEVICE)

    pos_weight = getattr(train_loader.dataset, "pos_weight", None)
    if pos_weight is not None:
        pos_weight = pos_weight.to(c.DEVICE, non_blocking=True)

    for epoch in range(c.EPOCHS):
        model.train()
        running_loss = 0.0
        running_bce = 0.0
        running_triplet = 0.0

        for inputs, labels, _, _ in tqdm(
            train_loader, desc=f"Training Epoch {epoch + 1}/{c.EPOCHS}"
        ):

            # Adjust learning rate for this epoch
            prog = epoch / (c.EPOCHS - 1) if c.EPOCHS > 1 else 0.0
            learn_rate = c.LEARNING_RATE * (c.FINAL_LR_MULTIPLE**prog)
            for param_group in optimizer.param_groups:
                param_group["lr"] = learn_rate

            inputs = inputs.to(c.DEVICE, non_blocking=True)
            labels = labels.to(c.DEVICE, non_blocking=True)

            optimizer.zero_grad()
            outputs, features = model.forward_features(inputs)

            loss, components = combined_bce_triplet_loss(
                logits=outputs,
                labels=labels,
                embeddings=features,
                lambda_bce=c.LAMBDA_BCE,
                lambda_tri=c.LAMBDA_TRIPLET if c.OSR_ENABLED else 0.0,
                margin=c.TRIPLET_MARGIN,
                pos_weight=pos_weight,
                hard_positive_quantile=c.TRIPLET_HARD_POS_QUANTILE,
                hard_negative_quantile=c.TRIPLET_HARD_NEG_QUANTILE,
            )

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_bce += components["bce"].item()
            running_triplet += components["triplet"].item()

        print(
            f"Epoch {epoch + 1}/{c.EPOCHS}, "
            f"Loss: {running_loss / len(train_loader):.4f} "
            f"(BCE: {running_bce / len(train_loader):.4f}, "
            f"Triplet: {running_triplet / len(train_loader):.4f})"
        )

    return model


def _categorize_overlap(num_known: int, num_unknown: int) -> str:
    total = num_known + num_unknown
    if total == 1:
        return "Known Modulation" if num_known == 1 else "Unknown Modulation"
    if total == 2:
        if num_known == 2:
            return "Known + Known"
        if num_known == 1:
            return "Known + Unknown"
        return "Unknown + Unknown"
    if num_unknown == 0:
        return f"{num_known} Known"
    if num_known == 0:
        return f"{num_unknown} Unknown"
    return f"{num_known} Known + {num_unknown} Unknown"


def _is_prediction_absolute_correct(
    predicted_row: torch.Tensor, label_row: torch.Tensor, unknown_mods: List[str]
) -> bool:
    gt_known = set(label_row.nonzero(as_tuple=True)[0].tolist())
    pred_known = set(predicted_row.nonzero(as_tuple=True)[0].tolist())

    num_unknown = len(unknown_mods)
    num_known = len(gt_known)

    if num_unknown == 0:
        # Known-only transmission: exact known-label match is required.
        expected_known = gt_known
    elif num_known == 1 and num_unknown == 1:
        # Two-transmitter KNOWN+UNKNOWN is treated as UNKNOWN.
        expected_known = set()
    else:
        # Keep the existing behavior for all other unknown-containing cases.
        expected_known = set()

    return pred_known == expected_known


def _superclass_label(mod_names: List[str]) -> str:
    if not mod_names:
        return "UNKNOWN"
    unknowns = [name for name in mod_names if name == "UNKNOWN"]
    knowns = sorted(name for name in mod_names if name != "UNKNOWN")
    return "+".join(unknowns + knowns)


def _label_contains_unknown(label: str) -> bool:
    parts = [part.strip() for part in label.split("+") if part.strip()]
    return any(part == "UNKNOWN" for part in parts)


def _normalize_confusion_label(label: str, collapse_unknown: bool) -> str:
    if not collapse_unknown:
        return label
    if label.startswith(">"):
        return label
    if _label_contains_unknown(label):
        return c.CONTAINS_UNKNOWN_LABEL
    return label


def _build_superclass_label(
    known_names: List[str],
    unknown_count: int,
    transmitter_count: Optional[int] = None,
) -> str:
    names = list(known_names)
    if unknown_count > 0:
        names.extend(["UNKNOWN"] * unknown_count)
    if transmitter_count is not None and len(names) < transmitter_count:
        names.extend(["UNKNOWN"] * (transmitter_count - len(names)))
    return _superclass_label(names)


def _build_predicted_superclass_label(
    predicted_row: torch.Tensor,
    label_names: List[str],
    max_transmitters: Optional[int] = None,
    allow_unknown_label: bool = True,
) -> str:
    pred_idx = predicted_row.nonzero(as_tuple=True)[0].tolist()
    if not pred_idx:
        return "UNKNOWN" if allow_unknown_label else "NO_PREDICTION"
    if max_transmitters is not None and len(pred_idx) > max_transmitters:
        return f">{max_transmitters} Predicted GT Classes"
    pred_names = [label_names[j] for j in pred_idx]
    return _superclass_label(pred_names)


def _test_model(
    model: nn.Module,
    test_loader: DataLoader,
    osr_detector: Optional[FeatureBasedOpenSetDetector] = None,
    osr_enabled: bool = True,
):
    model.eval()
    results = {}
    label_names = sorted(
        test_loader.dataset.label_to_idx, key=test_loader.dataset.label_to_idx.get
    )
    sample_pairs = {}
    unknown_lists = getattr(test_loader.dataset, "sample_unknown_mods", None)
    sample_idx = 0
    threshold_tensor = (
        torch.from_numpy(np.array([c.THRESHOLD_VALUE])).to(c.DEVICE).view(1, -1)
    )
    case_accuracy = {}
    confusion_matrix = {}
    overlap_counts = getattr(test_loader.dataset, "overlap_counts", None)
    max_transmitters = max(overlap_counts) + 1 if overlap_counts else None

    with torch.no_grad():
        for inputs, labels, snrs, overlaps in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(c.DEVICE, non_blocking=True)
            labels = labels.to(c.DEVICE, non_blocking=True)

            outputs, features = model.forward_features(inputs)
            probs = torch.sigmoid(outputs)
            predicted = (probs > threshold_tensor).float()
            batch_size = inputs.size(0)
            for i in range(batch_size):
                snr = int(snrs[i].item())
                overlap = int(overlaps[i].item())
                gt_unknown = []
                if unknown_lists is not None and sample_idx < len(unknown_lists):
                    gt_unknown = unknown_lists[sample_idx]
                report_unknown_mods = gt_unknown if osr_enabled else []
                num_known = int(labels[i].sum().item())
                num_unknown = len(report_unknown_mods)
                transmitter_count = num_known + num_unknown
                if osr_enabled and osr_detector is not None:
                    predicted[i] = osr_detector.filter_predictions(
                        features[i], predicted[i]
                    )

                # Record sample predictions for debugging
                if len(sample_pairs.setdefault(snr, [])) < c.SAMPLE_PRINT_COUNT:
                    gt_idx = labels[i].nonzero(as_tuple=True)[0].tolist()
                    pred_idx = predicted[i].nonzero(as_tuple=True)[0].tolist()
                    gt_names = [label_names[j] for j in gt_idx]
                    if not gt_names and report_unknown_mods:
                        if len(report_unknown_mods) == 1:
                            gt_names = [f"UNKNOWN ({report_unknown_mods[0]})"]
                        else:
                            joined = ", ".join(report_unknown_mods)
                            gt_names = [f"UNKNOWN ({joined})"]
                    pred_names = [label_names[j] for j in pred_idx]
                    if not pred_names:
                        pred_names = ["UNKNOWN" if osr_enabled else "NO_PREDICTION"]
                    sample_pairs[snr].append((overlap, gt_names, pred_names))
                res = results.setdefault(overlap, {}).setdefault(
                    snr, {"correct": 0, "total": 0}
                )
                is_correct = int(
                    _is_prediction_absolute_correct(
                        predicted[i], labels[i], report_unknown_mods
                    )
                )
                res["correct"] += is_correct
                res["total"] += 1

                category_label = _categorize_overlap(num_known, num_unknown)
                case_stats = (
                    case_accuracy.setdefault(transmitter_count, {})
                    .setdefault(category_label, {})
                    .setdefault(snr, {"correct": 0, "total": 0})
                )
                case_stats["correct"] += is_correct
                case_stats["total"] += 1
                gt_idx = labels[i].nonzero(as_tuple=True)[0].tolist()
                gt_names = [label_names[j] for j in gt_idx]
                gt_label = _normalize_confusion_label(
                    _build_superclass_label(gt_names, num_unknown),
                    c.CONFUSION_COLLAPSE_UNKNOWN,
                )
                pred_label = _normalize_confusion_label(
                    _build_predicted_superclass_label(
                        predicted[i],
                        label_names,
                        max_transmitters,
                        allow_unknown_label=osr_enabled,
                    ),
                    c.CONFUSION_COLLAPSE_UNKNOWN,
                )
                confusion_matrix.setdefault(gt_label, {}).setdefault(pred_label, 0)
                confusion_matrix[gt_label][pred_label] += 1
                sample_idx += 1
    return (
        results,
        sample_pairs,
        case_accuracy,
        confusion_matrix,
    )


def _print_results(
    results,
    sample_pairs,
):
    total_correct = 0
    total_count = 0
    for snr_dict in results.values():
        for stats in snr_dict.values():
            total_correct += stats["correct"]
            total_count += stats["total"]

    if total_count:
        overall_acc = total_correct / total_count * 100
        print(f"Overall Accuracy: {overall_acc:.2f}%")
    for overlap, snr_dict in sorted(results.items()):
        label = f"{overlap+1} Transmitters"
        overlap_correct = sum(stats["correct"] for stats in snr_dict.values())
        overlap_total = sum(stats["total"] for stats in snr_dict.values())
        print(f"{label}:")
        if overlap_total:
            overlap_acc = overlap_correct / overlap_total * 100
            print(f"  Overall: Accuracy {overlap_acc:.2f}%")
        for snr, stats in sorted(snr_dict.items()):
            acc = stats["correct"] / stats["total"] * 100
            print(f"  SNR {snr}: Accuracy {acc:.2f}%")

    if sample_pairs and c.SAMPLE_PRINT_COUNT > 0:
        print("\nSample predictions (ground truth vs predicted):")
        for snr in sorted(sample_pairs):
            print(f"SNR {snr}:")
            for idx, (overlap, gt, pred) in enumerate(
                sorted(sample_pairs[snr], key=lambda x: x[0]), 1
            ):
                print(f"  Sample {idx} (overlap {overlap}): True={gt}, Pred={pred}")


def _series_has_values(series: List[float]) -> bool:
    return any(not math.isnan(value) for value in series)


def _series_from_accuracy_stats(
    stats_dict: Dict[int, Dict[str, int]],
    snrs: List[int],
) -> List[float]:
    series: List[float] = []
    for snr in snrs:
        stat = stats_dict.get(snr)
        total = stat.get("total", 0) if stat else 0
        if not stat or total == 0:
            series.append(float("nan"))
        else:
            series.append(100.0 * stat.get("correct", 0) / total)
    return series


def _series_from_case_accuracy_ratio(
    category_stats: Dict[str, Dict[int, Dict[str, int]]],
    snrs: List[int],
) -> List[float]:
    series: List[float] = []
    for snr in snrs:
        snr_correct = 0
        snr_total = 0
        for snr_dict in category_stats.values():
            stat = snr_dict.get(snr)
            if not stat:
                continue
            snr_correct += stat.get("correct", 0)
            snr_total += stat.get("total", 0)
        if snr_total == 0:
            series.append(float("nan"))
        else:
            series.append(100.0 * snr_correct / snr_total)
    return series


def _load_previous_sota_series() -> Tuple[List[int], Dict[int, Dict[int, float]]]:
    snrs = list(PREVIOUS_SOTA_SNRS)
    previous_sota_series: Dict[int, Dict[int, float]] = {}
    for transmitter_count, accuracies in PREVIOUS_SOTA_ACCURACY.items():
        previous_sota_series[transmitter_count] = {
            # Support prior SOTA data provided either as ratio (0-1) or percent (0-100).
            snr: (float(np.clip(acc, 0.0, 1.0) * 100.0))
            for snr, acc in zip(snrs, accuracies)
        }
    return snrs, previous_sota_series


def _preferred_category_order(transmitter_count: int) -> List[str]:
    if transmitter_count == 1:
        return ["Known Modulation", "Unknown Modulation"]
    if transmitter_count == 2:
        return ["Known + Known", "Known + Unknown", "Unknown + Unknown"]
    if transmitter_count == 3:
        return ["3 Known", "2 Known + 1 Unknown", "1 Known + 2 Unknown", "3 Unknown"]
    return [
        f"{num_known} Known + {transmitter_count - num_known} Unknown"
        for num_known in range(transmitter_count, -1, -1)
    ]


def _ordered_category_labels(
    transmitter_count: int,
    category_stats: Dict[str, Dict[int, Dict[str, int]]],
) -> List[str]:
    preferred_order = _preferred_category_order(transmitter_count)
    ordered_labels: List[str] = []
    seen_labels = set()

    for label in preferred_order:
        if label in category_stats:
            ordered_labels.append(label)
            seen_labels.add(label)

    remaining_labels = sorted(
        label for label in category_stats if label not in seen_labels
    )
    ordered_labels.extend(remaining_labels)
    return ordered_labels


def _category_contains_unknown(category_label: str) -> bool:
    return "unknown" in category_label.lower()


def _plot_absolute_accuracy(
    case_accuracy: Dict[int, Dict[str, Dict[int, Dict[str, int]]]],
    output_dir: Path,
    compare_against_previous_sota: bool = False,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []

    plt.figure()

    if compare_against_previous_sota:
        snrs, previous_sota_series = _load_previous_sota_series()
        if not snrs:
            return saved_paths

        color_map = {
            1: "red",
            2: "blue",
            3: "green",
        }
        transmitter_labels = {
            1: "Single Modulated Signal",
            2: "Two Overlapped Signals",
            3: "Three Overlapped Signals",
        }
        transmitter_counts = sorted(
            set(case_accuracy.keys()).union(previous_sota_series.keys())
        )

        for transmitter_count in transmitter_counts:
            color = color_map.get(transmitter_count)
            if color is None:
                cmap = plt.get_cmap("tab20")
                color = cmap((transmitter_count - 1) % cmap.N)
            tx_label = transmitter_labels.get(
                transmitter_count,
                f"{transmitter_count} Signals",
            )

            if transmitter_count in case_accuracy:
                model_series = _series_from_case_accuracy_ratio(
                    case_accuracy[transmitter_count], snrs
                )
                if _series_has_values(model_series):
                    plt.plot(
                        snrs,
                        model_series,
                        marker="o",
                        linestyle="-",
                        color=color,
                        label=f"{tx_label} (Ours)",
                    )

            previous_stats = previous_sota_series.get(transmitter_count, {})
            previous_series = [previous_stats.get(snr, float("nan")) for snr in snrs]
            if _series_has_values(previous_series):
                plt.plot(
                    snrs,
                    previous_series,
                    marker="^",
                    linestyle="--",
                    color=color,
                    label=f"{tx_label} (HybridNet)",
                )
    else:
        snr_values = set()
        for category_stats in case_accuracy.values():
            for snr_dict in category_stats.values():
                snr_values.update(snr_dict.keys())
        snrs = sorted(snr_values)
        if not snrs:
            return saved_paths

        line_styles = ["-", "--", ":", "-.", (0, (3, 1, 1, 1)), (0, (5, 1))]
        transmitter_counts = sorted(case_accuracy)
        style_map = {
            transmitter_count: line_styles[(transmitter_count - 1) % len(line_styles)]
            for transmitter_count in transmitter_counts
        }

        all_categories: List[str] = []
        for transmitter_count in transmitter_counts:
            ordered_labels = _ordered_category_labels(
                transmitter_count, case_accuracy[transmitter_count]
            )
            for label in ordered_labels:
                if label not in all_categories:
                    all_categories.append(label)

        cmap = plt.get_cmap("tab20")
        category_colors = {
            category: cmap(idx % cmap.N) for idx, category in enumerate(all_categories)
        }
        legend_categories_by_tx: Dict[int, List[str]] = {
            transmitter_count: [] for transmitter_count in transmitter_counts
        }

        for transmitter_count in transmitter_counts:
            category_stats = case_accuracy[transmitter_count]
            ordered_labels = _ordered_category_labels(transmitter_count, category_stats)
            for category_label in ordered_labels:
                snr_dict = category_stats[category_label]
                series = _series_from_accuracy_stats(snr_dict, snrs)
                if not _series_has_values(series):
                    continue
                marker_style = (
                    "x" if _category_contains_unknown(category_label) else "o"
                )
                plt.plot(
                    snrs,
                    series,
                    marker=marker_style,
                    linestyle=style_map[transmitter_count],
                    color=category_colors[category_label],
                )
                legend_categories_by_tx[transmitter_count].append(category_label)

        legend_handles: List[Line2D] = []
        legend_labels: List[str] = []
        heading_indices: List[int] = []
        for transmitter_count in transmitter_counts:
            tx_categories = legend_categories_by_tx.get(transmitter_count, [])
            if not tx_categories:
                continue
            heading_indices.append(len(legend_labels))
            tx_label = (
                f"{transmitter_count} Transmitter"
                if transmitter_count == 1
                else f"{transmitter_count} Transmitters"
            )
            legend_handles.append(
                Line2D([0], [0], linestyle="none", marker="", color="none")
            )
            legend_labels.append(tx_label)
            for category_label in tx_categories:
                marker_style = (
                    "x" if _category_contains_unknown(category_label) else "o"
                )
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker=marker_style,
                        linestyle=style_map[transmitter_count],
                        color=category_colors[category_label],
                    )
                )
                legend_labels.append(f"  {category_label}")

        if legend_handles:
            legend = plt.legend(
                legend_handles,
                legend_labels,
                bbox_to_anchor=(1.01, 1),
                loc="upper left",
                borderaxespad=0.0,
            )
            for idx in heading_indices:
                legend.get_texts()[idx].set_weight("bold")

    if len(plt.gca().lines) == 0:
        plt.close()
        return saved_paths

    # Save as a PDF for better vector quality in publications
    plt.title("Total Accuracy Against AWGN Signal-to-Noise Ratio")
    plt.xlabel("AWGN Signal-to-Noise Ratio (dB)")
    plt.ylabel("Total Accuracy (%)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.xticks(snrs)
    plt.ylim(0, 100)
    plt.xlim(min(snrs), max(snrs))

    if compare_against_previous_sota:
        plt.legend(loc="lower right")
        plt.gcf().set_size_inches(6, 6)
        plt.tight_layout()
    else:
        plt.gcf().set_size_inches(9, 6)
        box = plt.gca().get_position()
        plt.gca().set_position([box.x0, box.y0, box.width * 0.75, box.height])

    file_path = output_dir / "total_accuracy_all_tx.png"
    plt.savefig(file_path, dpi=1000, bbox_inches="tight")

    # Save as a PDF for better vector quality in publications
    pdf_path = output_dir / "total_accuracy_all_tx.pdf"
    plt.savefig(pdf_path, bbox_inches="tight")

    plt.close()
    saved_paths.append(file_path)
    return saved_paths


def _sort_confusion_labels(labels: List[str]) -> List[str]:
    def _overlap_count(label: str) -> int:
        return len([part.strip() for part in label.split("+") if part.strip()])

    def _sort_key(label: str) -> Tuple[int, int, str]:
        # Keep unknown-containing buckets at the bottom while preserving
        # deterministic ordering for all other labels.
        contains_unknown = label == c.CONTAINS_UNKNOWN_LABEL or _label_contains_unknown(
            label
        )
        overflow_prediction = label.startswith(">")
        bucket_rank = 2 if contains_unknown else (1 if overflow_prediction else 0)
        return (bucket_rank, _overlap_count(label), label)

    return sorted(
        labels,
        key=_sort_key,
    )


def _expected_predicted_label_for_gt(gt_label: str, collapse_unknown: bool) -> str:
    parts = [part.strip() for part in gt_label.split("+") if part.strip()]
    unknown_count = sum(1 for part in parts if part == "UNKNOWN")
    known_parts = [part for part in parts if part != "UNKNOWN"]

    # For the two-transmitter KNOWN+UNKNOWN case, the expected prediction is UNKNOWN.
    if len(parts) == 2 and len(known_parts) == 1 and unknown_count == 1:
        return _normalize_confusion_label("UNKNOWN", collapse_unknown)

    return _normalize_confusion_label(_superclass_label(known_parts), collapse_unknown)


def _plot_confusion_matrix(
    confusion_matrix: Dict[str, Dict[str, int]],
    output_dir: Path,
) -> Optional[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    gt_labels = [
        label for label in confusion_matrix.keys() if not label.startswith(">")
    ]
    pred_label_set = set()
    for pred_dict in confusion_matrix.values():
        pred_label_set.update(pred_dict.keys())
    pred_labels = _sort_confusion_labels(list(pred_label_set))
    gt_labels = _sort_confusion_labels(gt_labels)
    if not gt_labels or not pred_labels:
        return None

    gt_index = {label: idx for idx, label in enumerate(gt_labels)}
    pred_index = {label: idx for idx, label in enumerate(pred_labels)}
    cm = np.zeros((len(gt_labels), len(pred_labels)), dtype=int)
    for gt_label, pred_dict in confusion_matrix.items():
        if gt_label not in gt_index:
            continue
        gt_idx = gt_index[gt_label]
        for pred_label, count in pred_dict.items():
            if pred_label not in pred_index:
                continue
            cm[gt_idx, pred_index[pred_label]] += count

    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_percent = (
            np.divide(
                cm,
                row_sums,
                out=np.zeros_like(cm, dtype=float),
                where=row_sums != 0,
            )
            * 100.0
        )

    fig, ax = plt.subplots(
        figsize=(max(5, 0.4 * len(pred_labels)), max(5, 0.4 * len(gt_labels)))
    )
    # Use aspect="auto" so the heatmap fills the axes and avoids vertical padding.
    ax.imshow(
        cm_percent,
        interpolation="nearest",
        cmap=plt.cm.Blues,
        aspect="auto",
    )
    ax.set_title("Confusion Matrix")
    ax.set_xticks(np.arange(len(pred_labels)))
    ax.set_xticklabels(pred_labels, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(gt_labels)))
    ax.set_yticklabels(gt_labels)
    ax.set_ylabel("Ground Truth")
    ax.set_xlabel("Predicted")

    threshold = cm_percent.max() / 2.0 if cm_percent.size else 0
    expected_pred_idx = [
        pred_index.get(
            _expected_predicted_label_for_gt(
                label,
                c.CONFUSION_COLLAPSE_UNKNOWN,
            )
        )
        for label in gt_labels
    ]
    for i in range(cm_percent.shape[0]):
        for j in range(cm_percent.shape[1]):
            is_expected = expected_pred_idx[i] == j
            ax.text(
                j,
                i,
                f"{cm_percent[i, j]:.1f}",
                ha="center",
                va="center",
                color="white" if cm_percent[i, j] > threshold else "black",
                fontweight="bold" if is_expected else "normal",
                fontsize=8,
            )

    fig.tight_layout()
    file_path = output_dir / "confusion_matrix_all_tx.png"
    fig.savefig(file_path, dpi=1000)

    # Save as a PDF for better vector quality in publications
    pdf_path = output_dir / "confusion_matrix_all_tx.pdf"
    fig.savefig(pdf_path)

    plt.close(fig)
    return file_path


def main(train_dir: str, test_dir: str):
    """
    Main function for training and testing the model.
    """

    # Print the configuration settings
    c.print_config()

    # Print the dataset paths being used
    print("Using Datasets:")
    print("\tTraining Data:", train_dir)
    print("\tTesting Data:", test_dir)

    train_dataset = ModulationDataset(train_dir, transform=None, shuffle=True)
    test_dataset = ModulationDataset(
        test_dir,
        transform=None,
        shuffle=False,
        label_to_idx=train_dataset.label_to_idx,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=c.BATCH_SIZE,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=c.BATCH_SIZE,
        num_workers=0,
        pin_memory=True,
    )

    # Model, Loss, Optimizer
    num_classes = train_dataset.num_classes
    model = ModulationClassifier(num_classes)

    # 1: Train the model
    model = _train_model(train_loader, model)

    # 2: Optionally calibrate the feature-based open-set detector.
    osr_detector = None
    if c.OSR_ENABLED:
        osr_detector = FeatureBasedOpenSetDetector(quantile=c.OSR_QUANTILE)
        osr_detector.fit(model, train_loader, c.DEVICE)

        print("\nCalibrated open-set thresholds (feature-space Mahalanobis):")
        label_names = sorted(
            train_dataset.label_to_idx, key=train_dataset.label_to_idx.get
        )
        for superclass_key, stats in sorted(
            osr_detector.get_class_statistics().items()
        ):
            threshold = stats["threshold"]
            if superclass_key:
                superclass_name = "+".join(label_names[idx] for idx in superclass_key)
            else:
                superclass_name = "UNKNOWN"
            print(
                f"  Superclass [{superclass_name}]: "
                f"distance@{c.OSR_QUANTILE:.3f}={threshold:.4f}"
            )

    # 3: Test the model using calibrated thresholds
    (
        results,
        sample_pairs,
        case_accuracy,
        confusion_matrices,
    ) = _test_model(
        model,
        test_loader,
        osr_detector=osr_detector,
        osr_enabled=c.OSR_ENABLED,
    )

    # 4: Print results
    _print_results(results, sample_pairs)

    # 5: Plot accuracy graphs
    gen_uuid = str(uuid.uuid4())
    output_dir = Path(f"/home/s2062378/OutputFiles/{gen_uuid}")
    print("\nSaving accuracy plot to:", output_dir)

    # Plot the absolute accuracy
    _plot_absolute_accuracy(
        case_accuracy,
        output_dir,
        compare_against_previous_sota=(
            c.COMPARE_AGAINST_PREVIOUS_SOTA and not c.OSR_ENABLED
        ),
    )

    # Plot the confusion matrices
    _plot_confusion_matrix(confusion_matrices, output_dir)


def run_pipeline(
    train_dir: str,
    test_dir: str,
    rng_seed: int,
):
    """
    Library-friendly entry point. Keeps the old main() behaviour intact.
    """
    set_rng_seed(rng_seed)
    return main(train_dir, test_dir)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parses the command line arguments if entered through CLI.

    Args:
        argv (Optional[Sequence[str]], optional): Command line arguments. Defaults to None.

    Returns:
        argparse.Namespace: Parsed arguments from the command line.
    """
    parser = argparse.ArgumentParser(
        description="Train and evaluate the CNN-Model for modulation classification."
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        required=True,
        help="Path to the directory containing the training dataset.",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        required=True,
        help="Path to the directory containing the testing dataset.",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args(argv)


def cli(argv: Optional[Sequence[str]] = None) -> None:
    """Helper function if script is called through the CLI.

    Args:
        argv (Optional[Sequence[str]], optional): Command line arguments. Defaults to None.
    """
    args = parse_args(argv)
    run_pipeline(args.train_dir, args.test_dir, args.rng_seed)


if __name__ == "__main__":
    cli()
