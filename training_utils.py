import os
from typing import Union, Tuple, List, Dict, Optional, Callable, cast
from numpy.lib.shape_base import apply_along_axis

import pandas as pd

from torchvision import datasets
from common_utils import get_major_class_by_class_name

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from common_utils import AudioToMelPipe
from audiodataset import AudioDataset


# class HierachyAudioDataset(datasets.DatasetFolder):
#     """
#     root
#     - folder(A_2_012)
#         - data1.wav
#         - data2.wav
#     - folder(A_2_013)
#         - data3.wav
#         - data4.wav
#     - folder(A_2_014)
#         - ...

#     일 때, 'A' 를 클래스 라벨로 사용
#     """

#     @staticmethod
#     def find_classes_hierachy(directory):
#         """Finds the class folders in a dataset.

#         See :class:`DatasetFolder` for details.
#         """
#         classes = sorted(
#             entry.name for entry in os.scandir(directory) if entry.is_dir()
#         )
#         if not classes:
#             raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

#         classes_to_major_classes = {
#             cls_name: get_major_class_by_class_name(cls_name) for cls_name in classes
#         }

#         major_classes = sorted(
#             list(set([get_major_class_by_class_name(_class) for _class in classes]))
#         )
#         major_classes_to_idx = {cls_name: i for i, cls_name in enumerate(major_classes)}

#         class_to_idx = {
#             cls_name: major_classes_to_idx[classes_to_major_classes[cls_name]]
#             for i, cls_name in enumerate(classes)
#         }
#         return classes, class_to_idx

#     def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
#         return self._find_classes(directory)

#     def _find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
#         return HierachyAudioDataset.find_classes_hierachy(directory)


def get_dataset(
    data_path, phase, pipe_config, extensions=(".wav",), use_major_class=False
):
    """
    use_major_class : 대분류 사용 여부
    """
    datapipe = AudioToMelPipe(
        sample_rate=pipe_config.sample_rate,
        n_fft=pipe_config.n_fft,
        hop_length=pipe_config.hop_length,
        n_mels=pipe_config.n_mels,
        random_split=(phase == "train"),
    )

    target_frame_length = pipe_config.target_frame_length if phase != "test" else None

    return AudioDataset(
        f"{data_path}/{phase}/",
        loader=lambda x: datapipe.load_audio(
            x,
            target_frame_length=target_frame_length,
            min_audio_sample_length=pipe_config.min_audio_sample_length,
        ),
        transform=None,
        extensions=extensions,
        use_major_class=use_major_class,
    )


def extract_logs(
    lightning_logdir: str = "lightning_logs/version_0/",
    output_dir=".",
    target_tag: Union[str, None] = None,
):
    event_acc = EventAccumulator(lightning_logdir)
    event_acc.Reload()

    dfs = []
    if target_tag is None:
        for tag in event_acc.Tags()["scalars"]:
            if (
                tag.startswith("train")
                or tag.startswith("val")
                or tag.startswith("test")
            ):
                w_times, step_nums, vals = zip(*event_acc.Scalars(tag))
                dfs.append(pd.DataFrame(data=vals, index=step_nums, columns=[tag]))

    else:
        w_times, step_nums, vals = zip(*event_acc.Scalars(target_tag))
        dfs.append(pd.DataFrame(data=vals, index=step_nums, columns=[target_tag]))

    for df in dfs:
        df.to_csv(
            os.path.join(output_dir, f"{df.columns[0]}.csv"), index_label="step_nums"
        )

    return dfs
