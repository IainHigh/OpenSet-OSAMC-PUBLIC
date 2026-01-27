"""
config.py:
Configuration settings for CNN-Model training.
"""

# pylint: disable=import-error
from typing import Final
import torch


class Config:
    """Configuration settings for CNN-Model training."""

    def __init__(self):

        self.BATCH_SIZE: Final[int] = 256
        self.EPOCHS: Final[int] = 30
        self.LEARNING_RATE: Final[float] = 0.001
        self.FINAL_LR_MULTIPLE: Final[float] = 0.1
        self.RNG_SEED: Final[int] = 2050
        self.SAMPLE_PRINT_COUNT: Final[int] = (
            0  # Number of test samples to display for debugging.
        )

        # Loss function weights and parameters
        self.LAMBDA_BCE: Final[float] = 1.0
        self.LAMBDA_TRIPLET: Final[float] = 1.0
        self.TRIPLET_MARGIN: Final[float] = 1.0

        # Threshold for classification. If sigmoid(logit) > threshold, classify as present.
        self.THRESHOLD_VALUE: Final[float] = 0.5

        # Quantile used for Mahalanobis distance thresholds in OSR calibration.
        self.OSR_QUANTILE: Final[float] = 0.90

        self.DEVICE: Final[torch.device] = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        if not torch.cuda.is_available():
            print("WARNING: CUDA not available, using CPU which may be slow.")

    def print_config(self):
        """Print the current configuration settings."""
        print("Model Hyperparameters:")
        print(f"\tBATCH_SIZE: {self.BATCH_SIZE}")
        print(f"\tEPOCHS: {self.EPOCHS}")
        print(f"\tLEARNING_RATE: {self.LEARNING_RATE}")
        print(f"\tFINAL_LR_MULTIPLE: {self.FINAL_LR_MULTIPLE}")
        print(f"\tRNG_SEED: {self.RNG_SEED}")
        print(f"\tSAMPLE_PRINT_COUNT: {self.SAMPLE_PRINT_COUNT}")
        print(f"\tLAMBDA_BCE: {self.LAMBDA_BCE}")
        print(f"\tLAMBDA_TRIPLET: {self.LAMBDA_TRIPLET}")
        print(f"\tTRIPLET_MARGIN: {self.TRIPLET_MARGIN}")
        print(f"\tTHRESHOLD_VALUE: {self.THRESHOLD_VALUE}")
        print(f"\tOSR_QUANTILE: {self.OSR_QUANTILE}")
        print(f"\tDEVICE: {self.DEVICE}")
