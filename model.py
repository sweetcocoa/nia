import torch
import torch.nn as nn
from torchvision.models import resnet18, vgg16, mobilenet


class SimpleAudioClassificationModel(nn.Module):
    def __init__(self, num_classes, model="resnet18"):
        super().__init__()
        if model == "resnet18":
            self.model = resnet18(num_classes=num_classes)
            self.model.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )  # Change input channel to match channel dimensionality
        elif model == "vgg16":
            self.model = vgg16(num_classes=num_classes)
            self.model.features[0] = nn.Conv2d(
                1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            )
        elif model == "mobilenetv2":
            self.model = mobilenet.MobileNetV2(num_classes=num_classes)
            self.model.features[0][0] = nn.Conv2d(
                1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
            )
        else:
            raise ValueError(f"{model} is not allowed backbone network!!")

    def forward(self, x):
        return self.model(x)
