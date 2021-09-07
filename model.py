import torch
import torch.nn as nn
from torchvision.models import resnet18


class SimpleAudioClassificationModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = resnet18(num_classes=num_classes)
        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )  # Change input channel to match channel dimensionality

    def forward(self, x):
        return self.model(x)
