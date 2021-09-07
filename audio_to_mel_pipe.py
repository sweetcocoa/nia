import torchaudio
import torch
import torch.nn
import random


class AudioToMelPipe:
    def __init__(self, is_validation=True):
        self.melspectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=2048, hop_length=256, f_min=10.0, n_mels=128
        )
        self.is_validation = is_validation

    def load_audio(self, path, target_frame_length):
        y, sr = torchaudio.load(path, normalize=True)
        melspec = self.melspectrogram(y)
        if target_frame_length is None:
            return melspec
        else:
            return self.pad_melspec(melspec, target_frame_length)

    def pad_melspec(self, melspec, target_frame_length):
        if melspec.shape[-1] < target_frame_length:
            left_pad = (target_frame_length - melspec.shape[-1]) // 2
            right_pad = (target_frame_length - melspec.shape[-1]) - left_pad
            melspec = torch.nn.functional.pad(melspec, (left_pad, right_pad))
        elif melspec.shape[-1] > target_frame_length:
            if self.is_validation:
                offset = 0
            else:
                offset = random.randint(0, melspec.shape[-1] - target_frame_length - 1)
            melspec = melspec[..., offset : offset + target_frame_length]
        return melspec
