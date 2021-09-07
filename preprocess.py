import glob
import os
import shutil
import multiprocessing
from joblib import Parallel, delayed

import numpy as np
import librosa
import soundfile
from tqdm import tqdm
from omegaconf import OmegaConf

from .config import Config


def load_meta(meta_file):
    meta = OmegaConf.load(meta_file)
    return meta


def process_meta(meta_file, output):
    meta = load_meta(meta_file)

    # TODO 구버전 json 하위호환.. 공백 없는 것이 최종임.
    label_meta = (
        meta["Label data info"] if "Label data info" in meta else meta["LabelDataInfo"]
    )

    audio_file = meta_file.replace(".json", ".mp3")
    y, sr = librosa.load(audio_file, sr=Config.sample_rate, mono=True)

    # TODO : Train/Validation Split은 sample 기준으로 대략 8:2 지향, 어느 split에 속하는지는 파이썬 내장 hash함수 이용
    # 현재 : 그냥 파일명기준 클래스당 제일 먼저 오는 샘플파일 1개가 Validation set이 됨
    class_id = f"{label_meta.Division1}_{label_meta.Division2}_{label_meta.Class}"
    if os.path.exists(
        os.path.join(
            output,
            "val",
            class_id,
        )
    ):
        # if hash(os.path.basename(meta_file)) % 5 != 0:
        output_dir = os.path.join(
            output,
            "train",
            class_id,
        )
    else:
        output_dir = os.path.join(
            output,
            "val",
            class_id,
        )
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
    for class_id in os.listdir(os.path.join(output_dir, "val")):
        if class_id not in train_ids:
            shutil.rmtree(os.path.join(output_dir, "val", class_id))
            if verbose:
                print("Removed ", class_id)


def main(meta_dir, output_dir, verbose=True):
    meta_files = get_meta_files_from(meta_dir, verbose=verbose)
    assert len(meta_files) > 0, "Not enough metafiles"

    def meta_iterator():
        pbar = tqdm(meta_files)
        for meta_file in pbar:
            pbar.set_description(meta_file)
            yield meta_file, output_dir

    # TODO : data 늘어나면 data split 바꾸고 n_jobs 늘리기
    num_segments = Parallel(n_jobs=1)(
        delayed(process_meta)(meta_file, output_dir)
        for meta_file, output_dir in meta_iterator()
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
