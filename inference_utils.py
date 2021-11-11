import os
import matplotlib.pyplot as plt
from numpy.lib.arraysetops import isin
import seaborn as sn
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchmetrics
from audiodataset import AudioDataset

from common_utils import AudioToMelPipe, get_major_class_by_class_name
from training_utils import get_dataset
from model import SimpleAudioClassificationModel


class Inference:
    def __init__(
        self,
        checkpoint_path,
        config,
        device="cuda",
        class_to_idx=None,
        idx_to_class=None,
    ):

        pipe_config = config.pipe

        self.pipe = AudioToMelPipe(
            sample_rate=pipe_config.sample_rate,
            n_fft=pipe_config.n_fft,
            hop_length=pipe_config.hop_length,
            n_mels=pipe_config.n_mels,
            random_split=False,
        )

        if class_to_idx is None or idx_to_class is None:
            log_dir = os.path.dirname(os.path.dirname(checkpoint_path))
            self.class_to_idx = torch.load(os.path.join(log_dir, "class_to_idx.pth"))
            self.idx_to_class = torch.load(os.path.join(log_dir, "idx_to_class.pth"))
        else:
            if isinstance(class_to_idx, str):
                self.class_to_idx = torch.load(class_to_idx)
            else:
                self.class_to_idx = self.model.class_to_idx

            if isinstance(idx_to_class, str):
                self.idx_to_class = torch.load(idx_to_class)
            else:
                self.idx_to_class = self.model.idx_to_class

        self.model = self.load_checkpoint(checkpoint_path, config)
        self.min_audio_sample_length = pipe_config.min_audio_sample_length
        self.device = device
        self.model = self.model.to(self.device)

    def load_checkpoint(self, checkpoint_path, config):
        def build_weight_only(experiment_state_dict):
            model_state_dict = dict()
            for k, v in experiment_state_dict["state_dict"].items():
                model_state_dict[k[len("model.") :]] = v
            return model_state_dict

        model = SimpleAudioClassificationModel(
            len(self.class_to_idx), model=config.model
        )
        experiment_state_dict = torch.load(checkpoint_path)
        model_state_dict = build_weight_only(experiment_state_dict)
        model.load_state_dict(model_state_dict)
        model.eval()
        return model

    def inference_audio(self, audio_path):
        with torch.no_grad():
            melspectrogram = (
                self.pipe.load_audio(
                    audio_path,
                    target_frame_length=None,
                    min_audio_sample_length=self.min_audio_sample_length,
                )
                .unsqueeze(0)
                .to(self.device)
            )
            logits = self.model(melspectrogram)
            index = nn.functional.softmax(logits, dim=-1).argmax().item()
            probabilities = nn.functional.softmax(logits, dim=-1)

            predicted_label = self.idx_to_class[index]
            prob = probabilities.detach().cpu().numpy().ravel()[index]
        return predicted_label, prob

    def get_confusion_matrix_of_dataset(
        self, dataset_path, config, split="test", output_dir="."
    ):
        test_dataset = get_dataset(
            dataset_path,
            split,
            config.pipe,
            use_major_class=config.dataset.use_major_class,
        )
        dataloader_test = DataLoader(test_dataset, batch_size=1, num_workers=0)
        metric = torchmetrics.Accuracy(num_classes=len(test_dataset.class_to_idx))
        conf_normal = torchmetrics.ConfusionMatrix(
            len(test_dataset.class_to_idx), normalize="true"
        )

        conf = torchmetrics.ConfusionMatrix(
            len(test_dataset.class_to_idx), normalize=None
        )

        metric = metric.to(self.device)
        conf_normal = conf_normal.to(self.device)
        conf = conf.to(self.device)

        correct, total = 0, 0
        with torch.no_grad():
            for img, label in dataloader_test:
                img, label = img.to(self.device), label.to(self.device)
                pred = torch.softmax(self.model(img), dim=-1)
                correct += (pred.argmax(dim=-1) == label).sum()
                total += len(img)
                metric(pred, label)
                conf(pred, label)
                conf_normal(pred, label)
            metric.compute()
            conf.compute()
            conf_normal.compute()

        confusion_matrix = conf.compute().cpu().numpy()
        confusion_matrix_normal = conf_normal.compute().cpu().numpy()

        def save_confusion_matrix(
            confusion_matrix, index, output_dir, filename="confusion_matrix.jpg"
        ):
            df_cm = pd.DataFrame(confusion_matrix, index=index, columns=index)
            plt.figure(figsize=(30, 21))
            sn.heatmap(df_cm, annot=True)
            plt.savefig(os.path.join(output_dir, filename))

        save_confusion_matrix(
            confusion_matrix=confusion_matrix,
            index=list(test_dataset.class_to_idx.keys()),
            output_dir=output_dir,
            filename="confusion_matrix.jpg",
        )

        save_confusion_matrix(
            confusion_matrix=confusion_matrix_normal,
            index=list(test_dataset.class_to_idx.keys()),
            output_dir=output_dir,
            filename="confusion_matrix_normal.jpg",
        )

        return confusion_matrix


if __name__ == "__main__":
    from omegaconf import OmegaConf
    import argparse

    parser = argparse.ArgumentParser(description="checkpoint (.pth) path")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    # config = OmegaConf.load("config.yaml")
    # config.merge_with_cli()
    if args.checkpoint is None:
        print(
            """Usage:
python inference_utils.py checkpoint=/path/to/ckpt
"""
        )
        checkpoint = "lightning_logs/version_0/checkpoints/last.ckpt"
    else:
        checkpoint = args.checkpoint

    if args.config is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(checkpoint)), "config.yaml"
        )
        print(config_path)

    config = OmegaConf.load(config_path)

    inference = Inference(checkpoint, config, "cuda")

    output_dir = os.path.dirname(os.path.dirname(checkpoint))
    cm = inference.get_confusion_matrix_of_dataset(
        config.dataset.path, config, output_dir=output_dir
    )
    print(f"Confusion matrix Images can be seen at checkpoint's folder {output_dir})")
    print(cm)
