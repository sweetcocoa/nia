import torch
import torch.nn as nn
import torchaudio
from torchvision.models import resnet18, vgg16, mobilenet
from common_utils import LogMelSpectrogram


class MobileNetV2(nn.Module):
    def __init__(self, model_config, pipe_config):
        super().__init__()
        self.melspectrogram = LogMelSpectrogram(pipe_config)

        self.model = mobilenet.mobilenet_v2(pretrained=model_config.pretrained)

        self.model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.last_channel, model_config.num_classes),
        )

        # self.model = mobilenet.MobileNetV2(num_classes=num_classes)
        self.model.features[0][0] = nn.Conv2d(
            1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )

    def forward(self, x):
        X = self.melspectrogram(x)
        batch, freq, frame = X.shape
        X = X.view(batch, 1, freq, frame)
        return self.model(X)


def model(model_config, pipe_config):
    return MobileNetV2(model_config=model_config, pipe_config=pipe_config)
