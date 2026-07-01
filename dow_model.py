"""
dow_model.py  --  the DoWNet-style CNN (PyTorch)
================================================
Kept close to DoWNet for comparability (proposal section 3.4):
  - two conv blocks: 3x3 kernels, ReLU, batch-norm, max-pool
  - global average pooling on the final feature map
  - fully connected head with dropout p=0.5, softmax over 4 classes

`self.block2` is the last conv block -- Captum's Grad-CAM attaches to it.
"""
import random
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


class DoWNetCNN(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(                       # <- Grad-CAM target
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(32, n_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.gap(x).flatten(1)
        x = self.drop(x)
        return self.fc(x)
