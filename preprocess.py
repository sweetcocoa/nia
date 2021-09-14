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


def count_samples_per_class(meta_dir):
    data_count_dict = defaultdict(int)
    meta_files = get_meta_files_from(meta_dir, verbose=False)
    for meta_file in tqdm(meta_files):
        meta = OmegaConf.load(meta_file)
        class_id = get_class_id_by_meta(meta)
        data_count_dict[class_id] += 1
    return data_count_dict


def determine_split(
    meta_file, total_count, current_count, val_split=0.1, test_split=0.1
):
    meta = load_meta(meta_file)
    class_id = get_class_id_by_meta(meta)
    split = "train"
    total_number = total_count[class_id]
    current_number = current_count[class_id]
    val_index = max(1, total_number * val_split)
    test_index = val_index + max(1, total_number * test_split)

    if current_number < val_index:
        split = "val"
    elif current_number < test_index:
        split = "test"

    current_count[class_id] += 1
    return split


def process_meta(meta_file, output, split="val"):
    meta = load_meta(meta_file)
    class_id = get_class_id_by_meta(meta)
    label_meta = meta["LabelDataInfo"]
    audio_file = meta_file.replace(".json", ".mp3")

    warnings.filterwarnings("ignore")
    y, sr = librosa.load(audio_file, sr=16000, mono=True)

    # TODO : Train/Validation Split은 sample 기준으로 대략 8:2 지향, 어느 split에 속하는지는 파이썬 내장 hash함수 이용
    output_dir = os.path.join(output, split, class_id)
    os.makedirs(output_dir, exist_ok=True)

    for i, interval in enumerate(label_meta.Segmentations):
        output_path = os.path.join(
            output_dir,
            os.path.basename(audio_file).replace(".mp3", f"_{i}.wav"),
        )
        segment = y[int(interval[0] * sr) : int(interval[1] * sr)]
        soundfile.write(output_path, segment, samplerate=sr)
    num_segments = len(label_meta.Segmentations)
    return num_segments


def get_meta_files_from(meta_dir, verbose=True):
    meta_files = sorted(glob.glob(os.path.join(meta_dir, "**/*.json"), recursive=True))
    meta_files = list(filter(lambda x: x.find("label") == -1, meta_files))
    if verbose:
        print("# of json :", len(meta_files))
    return meta_files


def remove_invalid_class(output_dir, verbose=True):
    """
    개수가 적어서 validation에만 있는 클래스 데이터는 삭제.
    """
    train_ids = os.listdir(os.path.join(output_dir, "train"))
    valid_ids = os.listdir(os.path.join(output_dir, "val"))
    for class_id in valid_ids:
        if class_id not in train_ids:
            shutil.rmtree(os.path.join(output_dir, "val", class_id))
            if verbose:
                print("Removed ", class_id)

    if verbose:
        print("::Original::")
        print(f"train_ids {len(train_ids)}\nclass_ids {len(valid_ids)}")


def main(meta_dir, output_dir, verbose=True):
    meta_files = get_meta_files_from(meta_dir, verbose=verbose)
    assert len(meta_files) > 0, "Not enough metafiles"

    total_count = count_samples_per_class(meta_dir)
    current_count = defaultdict(int)
    if verbose:
        print("Samples per class :", total_count)

    def meta_iterator():
        pbar = tqdm(meta_files)
        for meta_file in pbar:
            pbar.set_description(meta_file)
            split = determine_split(
                meta_file, total_count=total_count, current_count=current_count
            )
            yield meta_file, output_dir, split

    # TODO : data 늘어나면 data split 바꾸고 n_jobs 늘리기
    num_segments = Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(process_meta)(meta_file, output_dir, split)
        for meta_file, output_dir, split in meta_iterator()
    )

    print("# of segments :", sum(num_segments))
    remove_invalid_class(output_dir, verbose=verbose)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    main(args.input, args.output)
