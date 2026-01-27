"""
classifier.py:
This module contains the implementation of the ModulationClassifier.
A deep learning model for classifying modulation types in wireless signals.
"""

# pylint: disable=import-error
# pylint: disable=too-few-public-methods
from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with two convolutional layers and a skip connection.

    Args:
        nn (Module): PyTorch neural network module.
    """

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(
            out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.down = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Conv1d(
                in_ch, out_ch, kernel_size=1, stride=stride, bias=False
            )

    def forward(self, x):
        """Forward pass for the residual block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, in_ch, L).

        Returns:
            torch.Tensor: Output tensor of shape (B, out_ch, L').
        """
        identity = self.down(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + identity)
        return x


class ModulationClassifier(nn.Module):
    """A deep learning model for classifying modulation types in wireless signals.

    Args:
        nn (Module): PyTorch neural network module.
    """

    def __init__(self, num_classes):
        super().__init__()
        # Stem: early downsampling to shrink sequence length quickly
        self.stem = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=7, stride=2, padding=3, bias=False),  # L/2
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2, bias=False),  # L/4
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        # Residual stages with stride-2 downsampling
        self.res1 = ResidualBlock(128, 256, stride=2)  # L/8
        self.res2 = ResidualBlock(256, 512, stride=2)  # L/16
        self.res3 = ResidualBlock(512, 512, stride=1)  # keep channels
        # Light projection before head
        self.proj = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )
        # Global Average Pooling, no giant flatten
        self.pool = nn.AdaptiveAvgPool1d(1)  # (B, 512, 1)
        self.flatten = nn.Flatten()  # (B, 512)
        self.feature = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward_features(self, x):
        """Forward pass for the modulation classifier.
        Return both logits and intermediate features for downstream tasks.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 2, L).

        Returns:
            torch.Tensor: Output tensor of shape (B, num_classes).
            torch.Tensor: Intermediate feature tensor of shape (B, 256).
        """

        x = self.stem(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.proj(x)
        x = self.pool(x)
        x = self.flatten(x)
        features = self.feature(x)
        logits = self.classifier(features)
        return logits, features
