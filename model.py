from time import process_time
import torch
import torch.nn as nn
import importlib


class SimpleAudioClassificationModel(nn.Module):
    def __init__(self, model_config, pipe_config):
        super().__init__()
        BackBone = importlib.import_module(
            "backbones." + model_config.name
        ).__getattribute__("model")
        self.model = BackBone(model_config=model_config, pipe_config=pipe_config)

    def forward(self, x):
        return self.model(x)
