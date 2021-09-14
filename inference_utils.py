import os
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchmetrics

from common_utils import AudioToMelPipe
from training_utils import get_dataset
from model import SimpleAudioClassificationModel


class Inference:
    def __init__(self, checkpoint_path, config):
        self.class_to_idx = Inference.get_class_to_idx(config)
        self.idx_to_class = [
            k[0]
            for k in sorted(
                [(k, v) for k, v in self.class_to_idx.items()], key=lambda x: x[1]
            )
        ]

        pipe_config = config.pipe

        self.pipe = AudioToMelPipe(
            sample_rate=pipe_config.sample_rate,
            n_fft=pipe_config.n_fft,
            hop_length=pipe_config.hop_length,
            n_mels=pipe_config.n_mels,
            random_split=False,
        )
        self.model = self.load_checkpoint(checkpoint_path)
        self.min_audio_sample_length = pipe_config.min_audio_sample_length

    @staticmethod
    def get_class_to_idx(config):
        dir = os.path.join(config.dataset.path, "val")
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return class_to_idx

    def load_checkpoint(self, checkpoint_path):
        def build_weight_only(experiment_state_dict):
            model_state_dict = dict()
            for k, v in experiment_state_dict["state_dict"].items():
                model_state_dict[k[len("model.") :]] = v
            return model_state_dict

        model = SimpleAudioClassificationModel(len(self.class_to_idx))
        experiment_state_dict = torch.load(checkpoint_path)
        model_state_dict = build_weight_only(experiment_state_dict)
        model.load_state_dict(model_state_dict)
        model.eval()
        return model

    def inference_audio(self, audio_path):
        with torch.no_grad():
            melspectrogram = self.pipe.load_audio(
                audio_path,
                target_frame_length=None,
                min_audio_sample_length=self.min_audio_sample_length,
            ).unsqueeze(0)
            logits = self.model(melspectrogram)
            index = nn.functional.softmax(logits, dim=-1).argmax().item()
            probabilities = nn.functional.softmax(logits, dim=-1)

            predicted_label = self.idx_to_class[index]
            prob = probabilities.detach().numpy().ravel()[index]
        return predicted_label, prob

    def get_confusion_matrix_of_dataset(
        self, dataset_path, config, split="test", output_dir="."
    ):
        test_dataset = get_dataset(dataset_path, split, config.pipe)
        dataloader_test = DataLoader(test_dataset, batch_size=1, num_workers=0)
        metric = torchmetrics.Accuracy(num_classes=len(test_dataset.class_to_idx))
        conf = torchmetrics.ConfusionMatrix(
            len(test_dataset.class_to_idx), normalize="true"
        )

        correct, total = 0, 0
        with torch.no_grad():
            for img, label in dataloader_test:
                pred = torch.softmax(self.model(img), dim=-1)
                correct += (pred.argmax(dim=-1) == label).sum()
                total += len(img)
                metric(pred, label)
                conf(pred, label)
            metric.compute()
            conf.compute()

        confusion_matrix = conf.compute().numpy()

        def save_confusion_matrix(confusion_matrix, index, output_dir):
            df_cm = pd.DataFrame(confusion_matrix, index=index, columns=index)
            plt.figure(figsize=(20, 14))
            sn.heatmap(df_cm, annot=True)
            plt.savefig(os.path.join(output_dir, "confusion_matrix.jpg"))

        save_confusion_matrix(
            confusion_matrix=confusion_matrix,
            index=list(test_dataset.class_to_idx.keys()),
            output_dir=output_dir,
        )
        return confusion_matrix


if __name__ == "__main__":
    from omegaconf import OmegaConf

    config = OmegaConf.load("config.yaml")
    config.merge_with_cli()
    if config.checkpoint is None:
        print("""Usage:
python inference_utils.py checkpoint=/path/to/ckpt
""")
        checkpoint = "lightning_logs/version_0/checkpoints/last.ckpt"
    else:
        checkpoint = config.checkpoint
    
    inference = Inference(checkpoint, config)

    cm = inference.get_confusion_matrix_of_dataset(config.dataset.path, config)
    print("Confusion matrix (Image can be seen at current folder)")
    print(cm)