import glob
import os
import shutil
import multiprocessing
import warnings
from joblib import Parallel, delayed
from collections import defaultdict

import numpy as np
import librosa
import soundfile
from tqdm import tqdm
from omegaconf import OmegaConf


def load_meta(meta_file):
    meta = OmegaConf.load(meta_file)
    return meta


def get_class_id_by_meta(meta):
    lcfg = meta["LabelDataInfo"]
    return f"{lcfg.Division1}_{lcfg.Division2}_{lcfg.Class}"


def get_segments_by_meta(meta):
    lcfg = meta["LabelDataInfo"]
    return lcfg.Segmentations


def count_samples_per_class(input_dir, metas=None):
    data_count_dict = defaultdict(int)
    if metas is None:
        meta_files = get_meta_files_from(input_dir, verbose=False)
        metas = load_metas(meta_files)

    for meta in tqdm(metas):
        class_id = get_class_id_by_meta(meta)
        data_count_dict[class_id] += 1
    return data_count_dict


def count_segments_per_class(input_dir, metas=None):
    data_count_dict = defaultdict(int)
    if metas is None:
        meta_files = get_meta_files_from(input_dir, verbose=False)
        metas = load_metas(meta_files)

    for meta in tqdm(metas):
        class_id = get_class_id_by_meta(meta)
        num_segments = len(get_segments_by_meta(meta))
        data_count_dict[class_id] += num_segments
    return data_count_dict


def determine_split(
    meta,
    total_sample_count,
    current_count,
    val_split=0.1,
    test_split=0.1,
    minimum_count=3,
):
    class_id = get_class_id_by_meta(meta)
    split = "train"
    total_number = total_sample_count[class_id]

    # not enough samples for split
    if total_number < minimum_count:
        current_count[class_id] += 1
        return "val"

    current_number = current_count[class_id]
    val_index = max(1, total_number * val_split)
    test_index = val_index + max(1, total_number * test_split)

    if current_number < val_index:
        split = "val"
    elif current_number < test_index:
        split = "test"

    current_count[class_id] += 1
    return split


def process_meta(meta, audio_file, output_dir, split="val", sample_rate=16000):
    class_id = get_class_id_by_meta(meta)
    label_meta = meta["LabelDataInfo"]

    warnings.filterwarnings("ignore")
    y, sr = librosa.load(audio_file, sr=sample_rate, mono=True)

    # TODO : Train/Validation Split??? sample ???????????? ?????? 8:2 ??????, ?????? split??? ??????????????? ????????? ?????? hash?????? ??????
    audio_output_dir = os.path.join(output_dir, split, class_id)
    os.makedirs(audio_output_dir, exist_ok=True)

    for i, interval in enumerate(label_meta.Segmentations):
        output_path = os.path.join(
            audio_output_dir,
            os.path.basename(audio_file).replace(".mp3", f"_{i}.wav"),
        )
        segment = y[int(interval[0] * sr) : int(interval[1] * sr)]
        soundfile.write(output_path, segment, samplerate=sr)
    num_segments = len(label_meta.Segmentations)
    return num_segments


def get_meta_files_from(input_dir, verbose=True):
    meta_files = sorted(glob.glob(os.path.join(input_dir, "**/*.json"), recursive=True))
    meta_files = list(filter(lambda x: x.find("label") == -1, meta_files))
    if verbose:
        print("# of json :", len(meta_files))
    return meta_files


def load_metas(meta_files):
    """
    Parallel module??? order??? ?????????.
    """
    metas = Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(load_meta)(meta_file) for meta_file in tqdm(meta_files)
    )
    return metas


def print_pretty_dict(dic):
    list_dic = sorted([(k, v) for k, v in dic.items()], key=lambda x: x[0])
    string = str()
    for k, v in list_dic:
        string += f"{k}: {v}\n"
    return string


def remove_invalid_class(output_dir, verbose=True):
    """
    ????????? ????????? validation?????? ?????? ????????? ???????????? ??????.
    """
    train_ids = os.listdir(os.path.join(output_dir, "train"))
    valid_ids = sorted(os.listdir(os.path.join(output_dir, "val")))
    if verbose:
        print("????????? id ??????")
    for class_id in valid_ids:
        if class_id not in train_ids:
            shutil.rmtree(os.path.join(output_dir, "val", class_id))
            if verbose:
                print(class_id)

    if verbose:
        print(f"??????????????????ids: {len(train_ids)}\n???????????????ids: {len(valid_ids)}")


def main(input_dir, output_dir, minimum_sample, sample_rate, verbose=True):
    meta_files = get_meta_files_from(input_dir, verbose=verbose)
    assert len(meta_files) > 0, "Not enough metafiles"
    metas = load_metas(meta_files)

    total_sample_count = count_samples_per_class(input_dir, metas=metas)
    total_segment_count = count_segments_per_class(input_dir, metas=metas)

    current_count = defaultdict(int)
    if verbose:
        print(
            "Samples per class", print_pretty_dict(dict(total_sample_count)), sep="\n"
        )
        print(
            "Segments per class", print_pretty_dict(dict(total_segment_count)), sep="\n"
        )

    def meta_iterator():
        pbar = tqdm(list(zip(metas, meta_files)))
        for meta, meta_file in pbar:
            pbar.set_description(meta_file)
            split = determine_split(
                meta=meta,
                total_sample_count=total_sample_count,
                current_count=current_count,
                minimum_count=minimum_sample,
            )

            audio_file = meta_file.replace(".json", ".mp3").replace("??????????????????", "???????????????")
            if not os.path.exists(audio_file):
                raise ValueError("Not exist : ", meta_file, audio_file)
            yield meta, audio_file, output_dir, split

    # TODO : data ???????????? data split ????????? n_jobs ?????????
    num_segments = Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(process_meta)(meta, audio_file, output_dir, split, sample_rate)
        for meta, audio_file, output_dir, split in meta_iterator()
    )

    print("# of segments :", sum(num_segments))
    remove_invalid_class(output_dir, verbose=verbose)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--minimum_sample", type=int, required=True)
    parser.add_argument("--sample_rate", type=int, required=True)
    args = parser.parse_args()
    main(**vars(args))
