import os
from typing import Union

import pandas as pd
from torchvision import datasets
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from common_utils import AudioToMelPipe


def get_dataset(data_path, phase, pipe_config, extensions=(".wav",)):
    datapipe = AudioToMelPipe(
        sample_rate=pipe_config.sample_rate,
        n_fft=pipe_config.n_fft,
        hop_length=pipe_config.hop_length,
        n_mels=pipe_config.n_mels,
        random_split=(phase == "train"),
    )

    target_frame_length = pipe_config.target_frame_length if phase != "test" else None
    return datasets.DatasetFolder(
        f"{data_path}/{phase}/",
        loader=lambda x: datapipe.load_audio(
            x,
            target_frame_length=target_frame_length,
            min_audio_sample_length=pipe_config.min_audio_sample_length,
        ),
        transform=None,
        extensions=extensions,
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
