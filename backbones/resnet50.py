import torch
import torch.nn as nn
import torchaudio
from torchvision.models import resnet50
from torchvision.models.resnet import BasicBlock, Bottleneck
from common_utils import LogMelSpectrogram


class ResNet50(nn.Module):
    def __init__(self, model_config, pipe_config):
        super().__init__()
        self.melspectrogram = LogMelSpectrogram(pipe_config)

        self.model = resnet50(pretrained=model_config.pretrained)
        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )  # Change input channel to match channel dimensionality
        self.model.fc = nn.Linear(512 * Bottleneck.expansion, model_config.num_classes)

    def forward(self, x):
        X = self.melspectrogram(x)
        batch, freq, frame = X.shape
        X = X.view(batch, 1, freq, frame)
        return self.model(X)


def model(model_config, pipe_config):
    return ResNet50(model_config=model_config, pipe_config=pipe_config)
