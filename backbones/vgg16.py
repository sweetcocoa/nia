import torch
import torch.nn as nn
import torchaudio
from torchvision.models import resnet18, vgg16, mobilenet
from common_utils import LogMelSpectrogram


class VGG16(nn.Module):
    def __init__(self, num_classes, pipe_config):
        super().__init__()
        self.melspectrogram = LogMelSpectrogram(pipe_config)
        self.model = vgg16(num_classes=num_classes)
        self.model.features[0] = nn.Conv2d(
            1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )

    def forward(self, x):
        X = self.melspectrogram(x)
        batch, freq, frame = X.shape
        X = X.view(batch, 1, freq, frame)
        return self.model(X)


def model(num_classes, pipe_config):
    return VGG16(num_classes=num_classes, pipe_config=pipe_config)
