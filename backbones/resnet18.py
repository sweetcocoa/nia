import torch
import torch.nn as nn
import torchaudio
from torchvision.models import resnet18, vgg16, mobilenet
from common_utils import LogMelSpectrogram


class ResNet18(nn.Module):
    def __init__(self, num_classes, pipe_config):
        super().__init__()
        self.melspectrogram = LogMelSpectrogram(pipe_config)
        self.model = resnet18(num_classes=num_classes)
        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )  # Change input channel to match channel dimensionality

    def forward(self, x):
        X = self.melspectrogram(x)
        batch, freq, frame = X.shape
        X = X.view(batch, 1, freq, frame)
        return self.model(X)


def model(num_classes, pipe_config):
    return ResNet18(num_classes=num_classes, pipe_config=pipe_config)
