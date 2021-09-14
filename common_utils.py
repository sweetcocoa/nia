import torchaudio
import torch
import torch.nn
import random


class AudioToMelPipe:
    def __init__(
        self,
        sample_rate=16000,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        random_split=True,
    ):
        self.melspectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            f_min=10.0,
            n_mels=n_mels,
        )
        self.random_split = random_split

    def load_audio(self, path, target_frame_length=None, min_audio_sample_length=1.024):
        y, sr = torchaudio.load(path, normalize=True)
        if y.shape[1] < min_audio_sample_length:
            y = torch.nn.functional.pad(y, (0, min_audio_sample_length - y.shape[1]))
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
            if not self.random_split:
                offset = 0
            else:
                offset = random.randint(0, melspec.shape[-1] - target_frame_length - 1)
            melspec = melspec[..., offset : offset + target_frame_length]
        return melspec
