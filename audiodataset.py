import torch
from torch.utils.data.dataset import Dataset
import os
import glob

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
