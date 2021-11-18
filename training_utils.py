import os
from typing import Union, Tuple, List, Dict, Optional, Callable, cast
from numpy.lib.shape_base import apply_along_axis

import pandas as pd

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from common_utils import audio_load
from audiodataset import AudioDataset


def get_dataset(
    data_path, phase, pipe_config, extensions=(".wav",), use_major_class=False
):
    """
    use_major_class : 대분류 사용 여부
    """

    target_audio_sample_length = (
        (pipe_config.target_audio_sample_length) if phase != "test" else None
    )

    return AudioDataset(
        f"{data_path}/{phase}/",
        loader=lambda x: audio_load(
            x,
            target_audio_sample_length=target_audio_sample_length,
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
