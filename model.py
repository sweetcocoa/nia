import torch
import torch.nn as nn
import importlib


class SimpleAudioClassificationModel(nn.Module):
    def __init__(self, num_classes, pipe_config, model="resnet18"):
        super().__init__()
        BackBone = importlib.import_module("backbones." + model).__getattribute__(
            "model"
        )
        self.model = BackBone(num_classes=num_classes, pipe_config=pipe_config)

    def forward(self, x):
        return self.model(x)
