import os
import glob
from collections import Counter

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import WeightedRandomSampler

from common_utils import get_major_class_by_class_name


class AudioDataset(Dataset):
    def __init__(
        self, root, loader, transform=None, extensions=(".wav",), use_major_class=False
    ) -> None:
        super().__init__()
        self.root = root
        self.loader = loader
        self.transform = transform

        self.paths = []
        for extension in extensions:
            audios = sorted(glob.glob(os.path.join(root, "**", "*" + extension)))
            self.paths.extend(audios)

        self.cls_names = AudioDataset.get_classes(
            self.paths, use_major_class=use_major_class
        )

        self.classes = sorted(list(set(self.cls_names)))

        self.class_to_idx = dict(
            [(cls_name, i) for i, cls_name in enumerate(self.classes)]
        )

        self.idx_to_class = dict(
            [(i, cls_name) for i, cls_name in enumerate(self.classes)]
        )

        self.labels = [self.class_to_idx[cls_name] for cls_name in self.cls_names]
        self.samples = [(self.paths[i], self.labels[i]) for i in range(len(self.paths))]

    def save_counter(self, path):
        counter = Counter(self.cls_names)
        with open(path, "w") as f:
            for k, v in sorted(counter.items(), key=lambda x: x[0]):
                f.write(f"{k}, {v}\n")

    @staticmethod
    def get_labels_from_path(path):
        """
        '~~~/path/to/A_0_001/audio.wav'
        -> 'A_0_001'
        """
        return path.split(os.sep)[-2]

    @staticmethod
    def get_classes(paths, use_major_class):
        labels = [AudioDataset.get_labels_from_path(path) for path in paths]
        if use_major_class:
            labels = [get_major_class_by_class_name(label) for label in labels]
        return labels

    def __getitem__(self, index):
        path, target = self.samples[index]

        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return (sample, target)

    def __len__(self):
        return len(self.samples)

    def get_weighted_sampler(self):
        counter = Counter(self.cls_names)
        # TODO : smoothing??
        for k, v in counter.items():
            counter[k] = min(1500, v)
            # counter[k] = math.log2(v)
        weights = [1.0 / counter[k] for k in self.cls_names]
        weighted_sampler = WeightedRandomSampler(weights, num_samples=len(self))
        return weighted_sampler
