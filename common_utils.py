from threading import main_thread
from torch.nn.modules import normalization
import torchaudio
import torch
import torch.nn as nn
import random


class LogMelSpectrogram(nn.Module):
    def __init__(self, pipe_config) -> None:
        super().__init__()
        self.melspectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=pipe_config.sample_rate,
            n_fft=pipe_config.n_fft,
            hop_length=pipe_config.hop_length,
            f_min=10.0,
            n_mels=pipe_config.n_mels,
        )
        self.use_log = pipe_config.use_log

    def forward(self, x):
        # x : audio(batch, sample)
        # X : melspec (batch, freq, frame)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                X = self.melspectrogram(x)
                if self.use_log:
                    X = X.clamp(min=1e-6).log()
        return X


def audio_load(
    path,
    target_audio_sample_length=None,
    min_audio_sample_length=16384,
    random_split=False,
):
    # target_frame_length = audio_length // hop_size + 1
    # audio_length = (target_frame_length - 1) * hop_size

    y, sr = torchaudio.load(path, normalize=True)
    y = y[0]
    if y.shape[0] < min_audio_sample_length:
        y = torch.nn.functional.pad(y, (0, min_audio_sample_length - y.shape[0]))

    if target_audio_sample_length is None:
        return y
    else:
        if random_split:
            offset = random.randint(0, y.shape[-1] - target_audio_sample_length - 1)
        else:
            offset = 0
        y = y[offset : offset + target_audio_sample_length]

    return y


def get_major_class_by_class_name(cls_name):
    "A_1_2 -> A"
    return cls_name.split("_")[0]
