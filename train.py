import random
import sys
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.models import resnet18

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from flash.core.classification import ClassificationTask

from model import SimpleAudioClassificationModel
from audio_to_mel_pipe import AudioToMelPipe


def main(args):
    def get_dataset(phase, target_frame_length, extensions=(".wav",)):
        datapipe = AudioToMelPipe(is_validation=(phase == "val"))
        return datasets.DatasetFolder(
            f"{args.data_path}/{phase}/",
            loader=lambda x: datapipe.load_audio(
                x, target_frame_length=target_frame_length
            ),
            transform=None,
            extensions=extensions,
        )

    dataset_train = get_dataset(phase="train", target_frame_length=64)
    dataset_val = get_dataset(phase="val", target_frame_length=64)
    dataset_test = get_dataset(phase="val", target_frame_length=None)

    dataloader_train = DataLoader(
        dataset_train, batch_size=24, shuffle=True, num_workers=4
    )
    dataloader_val = DataLoader(dataset_val, batch_size=24, num_workers=4)
    dataloader_test = DataLoader(dataset_test, batch_size=1, num_workers=0)

    ckpt_callback = ModelCheckpoint(
        monitor="val_accuracy",
        filename="model-{epoch:04d}-{val_accuracy:.3f}",
        save_top_k=1,
        mode="max",
        save_last=True,
    )

    model = SimpleAudioClassificationModel(num_classes=len(dataset_val.classes))

    classifier = ClassificationTask(
        model,
        loss_fn=nn.functional.cross_entropy,
        optimizer=optim.Adam,
        learning_rate=1e-3,
    )

    trainer = pl.Trainer(
        max_epochs=100,
        gpus=1,
        log_every_n_steps=5,
        callbacks=ckpt_callback,
    )

    trainer.fit(
        classifier, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val
    )

    results = trainer.test(classifier, test_dataloaders=dataloader_test)

    print(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()
    main(args)
