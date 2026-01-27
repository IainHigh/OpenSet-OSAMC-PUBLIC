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
from typing import Dict, List, Optional

# Third Party Imports
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch import optim, nn
from tqdm import tqdm
import numpy as np

# Local Imports
from dataset import ModulationDataset
from classifier import ModulationClassifier
from open_set import FeatureBasedOpenSetDetector
from loss import combined_bce_triplet_loss
from configs import Config

c = Config()
matplotlib.use("Agg")


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
                lambda_tri=c.LAMBDA_TRIPLET,
                margin=c.TRIPLET_MARGIN,
                pos_weight=pos_weight,
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
        return "known" if num_known == 1 else "unknown"
    if total == 2:
        if num_known == 2:
            return "known + known"
        if num_known == 1:
            return "known + unknown"
        return "unknown + unknown"
    return f"{num_known} known + {num_unknown} unknown"


def _is_prediction_absolute_correct(
    predicted_row: torch.Tensor, label_row: torch.Tensor, unknown_mods: List[str]
) -> bool:
    gt_known = set(label_row.nonzero(as_tuple=True)[0].tolist())
    pred_known = set(predicted_row.nonzero(as_tuple=True)[0].tolist())

    if gt_known:
        expected_known = gt_known
    else:
        # Unknown-only transmissions should not trigger any known predictions.
        assert unknown_mods != [], "Unknown mods list should not be empty here."
        expected_known = set()

    return pred_known == expected_known


def _superclass_label(mod_names: List[str]) -> str:
    if not mod_names:
        return "UNKNOWN"
    unknowns = [name for name in mod_names if name == "UNKNOWN"]
    knowns = sorted(name for name in mod_names if name != "UNKNOWN")
    return "+".join(unknowns + knowns)


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
) -> str:
    pred_idx = predicted_row.nonzero(as_tuple=True)[0].tolist()
    if not pred_idx:
        return "UNKNOWN"
    if max_transmitters is not None and len(pred_idx) > max_transmitters:
        return f">{max_transmitters} Predicted GT Classes"
    pred_names = [label_names[j] for j in pred_idx]
    return _superclass_label(pred_names)


def _test_model(
    model: nn.Module,
    test_loader: DataLoader,
    osr_detector: Optional[FeatureBasedOpenSetDetector] = None,
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
                num_known = int(labels[i].sum().item())
                num_unknown = len(gt_unknown)
                transmitter_count = num_known + num_unknown
                if osr_detector is not None:
                    predicted[i] = osr_detector.filter_predictions(
                        features[i], predicted[i]
                    )

                # Record sample predictions for debugging
                if len(sample_pairs.setdefault(snr, [])) < c.SAMPLE_PRINT_COUNT:
                    gt_idx = labels[i].nonzero(as_tuple=True)[0].tolist()
                    pred_idx = predicted[i].nonzero(as_tuple=True)[0].tolist()
                    gt_names = [label_names[j] for j in gt_idx]
                    if not gt_names and gt_unknown:
                        if len(gt_unknown) == 1:
                            gt_names = [f"UNKNOWN ({gt_unknown[0]})"]
                        else:
                            joined = ", ".join(gt_unknown)
                            gt_names = [f"UNKNOWN ({joined})"]
                    pred_names = [label_names[j] for j in pred_idx]
                    if not pred_names:
                        pred_names = ["UNKNOWN"]
                    sample_pairs[snr].append((overlap, gt_names, pred_names))
                res = results.setdefault(overlap, {}).setdefault(
                    snr, {"correct": 0, "total": 0}
                )
                is_correct = int(
                    _is_prediction_absolute_correct(predicted[i], labels[i], gt_unknown)
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
                gt_label = _build_superclass_label(gt_names, num_unknown)
                pred_label = _build_predicted_superclass_label(
                    predicted[i], label_names, max_transmitters
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


def _plot_absolute_accuracy(
    case_accuracy: Dict[int, Dict[str, Dict[int, Dict[str, int]]]],
    output_dir: Path,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []

    category_order = {
        1: ["known", "unknown"],
        2: ["known + known", "known + unknown", "unknown + unknown"],
        3: [
            "3 known",
            "2 known + unknown",
            "known + 2 unknown",
            "3 unknown",
        ],
    }

    for transmitter_count, category_stats in sorted(case_accuracy.items()):
        snr_values = set()
        for snr_dict in category_stats.values():
            snr_values.update(snr_dict.keys())
        snrs = sorted(snr_values)

        plt.figure()
        preferred_order = category_order.get(transmitter_count, [])
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

        for category_label in ordered_labels:
            snr_dict = category_stats[category_label]
            series = _series_from_accuracy_stats(snr_dict, snrs)
            if not _series_has_values(series):
                continue
            plt.plot(snrs, series, marker="o", label=category_label)

        if len(plt.gca().lines) == 0:
            plt.close()
            continue
        tx_suffix = "transmitter" if transmitter_count == 1 else "transmitters"
        plt.title(
            f"Total Accuracy Against AWGN Signal-to-Noise Ratio ({transmitter_count} {tx_suffix})"
        )
        plt.xlabel("AWGN Signal-to-Noise Ratio (dB)")
        plt.ylabel("Total Accuracy (%)")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.xticks(snrs)
        plt.ylim(0, 100)
        plt.legend()
        plt.tight_layout()

        file_path = output_dir / f"total_accuracy_{transmitter_count}_tx.png"
        plt.savefig(file_path, dpi=1000)
        plt.close()
        saved_paths.append(file_path)

    return saved_paths


def _sort_confusion_labels(labels: List[str]) -> List[str]:
    return sorted(
        labels,
        key=lambda label: (
            label.startswith(">"),
            label.startswith("UNKNOWN"),
            label,
        ),
    )


def _expected_predicted_label_for_gt(gt_label: str) -> str:
    parts = [part.strip() for part in gt_label.split("+") if part.strip()]
    known_parts = [part for part in parts if part != "UNKNOWN"]
    return _superclass_label(known_parts)


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

    plt.figure(figsize=(max(5, 0.4 * len(pred_labels)), max(5, 0.4 * len(gt_labels))))
    plt.imshow(cm_percent, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (Label Accuracy %)")
    x = plt.colorbar(label="Percentage (%)")
    x.remove()
    plt.xticks(np.arange(len(pred_labels)), pred_labels, rotation=45, ha="right")
    plt.yticks(np.arange(len(gt_labels)), gt_labels)
    plt.ylabel("Ground Truth")
    plt.xlabel("Predicted")

    threshold = cm_percent.max() / 2.0 if cm_percent.size else 0
    expected_pred_idx = [
        pred_index.get(_expected_predicted_label_for_gt(label)) for label in gt_labels
    ]
    for i in range(cm_percent.shape[0]):
        for j in range(cm_percent.shape[1]):
            is_expected = expected_pred_idx[i] == j
            plt.text(
                j,
                i,
                f"{cm_percent[i, j]:.1f}",
                ha="center",
                va="center",
                color="white" if cm_percent[i, j] > threshold else "black",
                fontweight="bold" if is_expected else "normal",
                fontsize=8,
            )

    plt.tight_layout()
    file_path = output_dir / "confusion_matrix_all_tx.png"
    plt.savefig(file_path, dpi=1000)
    plt.close()
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

    # 2: Calibrate the feature-based open-set detector on the training data
    osr_detector = FeatureBasedOpenSetDetector(quantile=c.OSR_QUANTILE)
    osr_detector.fit(model, train_loader, c.DEVICE)

    print("\nCalibrated open-set thresholds (feature-space Mahalanobis):")
    for class_idx, stats in sorted(osr_detector.get_class_statistics().items()):
        threshold = stats["threshold"]
        print(f"  Class {class_idx}: distance@{c.OSR_QUANTILE:.3f}={threshold:.4f}")

    # 3: Test the model using calibrated thresholds
    (
        results,
        sample_pairs,
        case_accuracy,
        confusion_matrices,
    ) = _test_model(model, test_loader, osr_detector=osr_detector)

    # 4: Print results
    _print_results(results, sample_pairs)

    # 5: Plot accuracy graphs
    gen_uuid = str(uuid.uuid4())
    output_dir = Path(f"/home/s2062378/OutputFiles/{gen_uuid}")
    print("\nSaving accuracy plot to:", output_dir)

    # Plot the absolute accuracy
    _plot_absolute_accuracy(case_accuracy, output_dir)

    # Plot the confusion matrices
    _plot_confusion_matrix(confusion_matrices, output_dir)


if __name__ == "__main__":
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
    args = parser.parse_args()

    np.random.seed(c.RNG_SEED)  # Set random seed for reproducibility
    torch.manual_seed(c.RNG_SEED)
    main(args.train_dir, args.test_dir)
