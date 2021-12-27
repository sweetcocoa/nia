import torch
import torch.nn as nn
from torchvision.ops.misc import ConvNormActivation

from torchvision.models import mobilenetv3
from common_utils import LogMelSpectrogram

from functools import partial


class MobileNetV3(nn.Module):
    def __init__(self, model_config, pipe_config):
        super().__init__()
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        self.melspectrogram = LogMelSpectrogram(pipe_config)
        self.model = mobilenetv3.mobilenet_v3_large(pretrained=model_config.pretrained)
        self.model.features[0] = ConvNormActivation(
            1,
            16,
            kernel_size=3,
            stride=2,
            norm_layer=norm_layer,
            activation_layer=nn.Hardswish,
        )

        in_features = self.model.classifier[-1].weight.shape[1]
        self.model.classifier[-1] = nn.Linear(in_features, model_config.num_classes)

    def forward(self, x):
        X = self.melspectrogram(x)
        batch, freq, frame = X.shape
        X = X.view(batch, 1, freq, frame)
        return self.model(X)


def model(model_config, pipe_config):
    return MobileNetV3(model_config=model_config, pipe_config=pipe_config)
