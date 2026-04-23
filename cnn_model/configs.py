"""
configs.py:
Configuration settings for CNN-Model training.
"""

# pylint: disable=import-error
import torch


class Config:
    """Configuration settings for CNN-Model training."""

    ####################
    # Model hyperparameters:
    ####################
    LEARNING_RATE = 0.001
    FINAL_LR_MULTIPLE = 0.1
    BATCH_SIZE = 2048  # (B)
    EPOCHS = 40
    # q: Quantile used for Mahalanobis distance thresholds in OSR calibration.
    OSR_QUANTILE = 0.95
    # T: Threshold for classification. If sigmoid(logit) > threshold, classify as present.
    THRESHOLD_VALUE = 0.5
    TRIPLET_MARGIN = 1.0  # Triplet Margin (m)

    # Quantiles for hard positive/negative [p, n]
    TRIPLET_HARD_POS_QUANTILE, TRIPLET_HARD_NEG_QUANTILE = 0.999, 0.001

    # Loss function weights
    LAMBDA_BCE, LAMBDA_TRIPLET = 1.0, 1.0

    ####################
    # Plotting settings:
    ####################

    # Enable open-set recognition components (triplet loss + Mahalanobis detector).
    OSR_ENABLED = True

    SAMPLE_PRINT_COUNT = 0  # Number of test samples to display for debugging.

    # Whether to include previous SOTA curves in evaluation plots.
    COMPARE_AGAINST_PREVIOUS_SOTA = False

    # Confusion-matrix label mode:
    # True -> collapse every label containing UNKNOWN into CONTAINS UNKNOWN.
    # False -> keep full UNKNOWN+KNOWN combinations.
    CONFUSION_COLLAPSE_UNKNOWN = True
    CONTAINS_UNKNOWN_LABEL = "contains_unknown"

    # Set device to GPU if available, otherwise fallback to CPU.
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, using CPU which may be slow.")

    def print_config(self):
        """Print the current configuration settings."""
        print("Model Hyperparameters:")
        print(f"\tBATCH_SIZE: {self.BATCH_SIZE}")
        print(f"\tEPOCHS: {self.EPOCHS}")
        print(f"\tLEARNING_RATE: {self.LEARNING_RATE}")
        print(f"\tFINAL_LR_MULTIPLE: {self.FINAL_LR_MULTIPLE}")
        print(f"\tSAMPLE_PRINT_COUNT: {self.SAMPLE_PRINT_COUNT}")
        print(f"\tLAMBDA_BCE: {self.LAMBDA_BCE}")
        print(f"\tLAMBDA_TRIPLET: {self.LAMBDA_TRIPLET}")
        print(f"\tTRIPLET_MARGIN: {self.TRIPLET_MARGIN}")
        print(f"\tTRIPLET_HARD_POS_QUANTILE: {self.TRIPLET_HARD_POS_QUANTILE}")
        print(f"\tTRIPLET_HARD_NEG_QUANTILE: {self.TRIPLET_HARD_NEG_QUANTILE}")
        print(f"\tTHRESHOLD_VALUE: {self.THRESHOLD_VALUE}")
        print(f"\tOSR_ENABLED: {self.OSR_ENABLED}")
        print(f"\tOSR_QUANTILE: {self.OSR_QUANTILE}")
        print(f"\tCONFUSION_COLLAPSE_UNKNOWN: {self.CONFUSION_COLLAPSE_UNKNOWN}")
        print(f"\tDEVICE: {self.DEVICE}")
