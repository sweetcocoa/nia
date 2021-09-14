import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf import OmegaConf
from flash.core.classification import ClassificationTask

from model import SimpleAudioClassificationModel
from training_utils import extract_logs, get_dataset


def main(config):

    dataset_train = get_dataset(
        config.dataset.path,
        phase="train",
        pipe_config=config.pipe,
    )
    dataset_val = get_dataset(
        config.dataset.path,
        phase="val",
        pipe_config=config.pipe,
    )
    dataset_test = get_dataset(
        config.dataset.path,
        phase="test",
        pipe_config=config.pipe,
    )

    print(
        f"""
Dataset Summary:
# of train : {len(dataset_train)},
# of val : {len(dataset_val)},
# of test : {len(dataset_test)},
"""
    )

    assert (
        len(dataset_train.classes)
        == len(dataset_val.classes)
        == len(dataset_test.classes)
    )

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
    )
    dataloader_test = DataLoader(dataset_test, batch_size=1, num_workers=0)

    ckpt_callback = ModelCheckpoint(
        monitor="val_f1",
        filename="model-{epoch:04d}-{val_f1:.3f}",
        save_top_k=1,
        mode="max",
        save_last=True,
    )

    model = SimpleAudioClassificationModel(num_classes=len(dataset_val.classes))

    # multi_label=True, num_class 가 specify됐을 때 f1 metric이 사용됨.
    classifier = ClassificationTask(
        model,
        loss_fn=nn.functional.cross_entropy,
        num_classes=len(dataset_val.classes),
        multi_label=True,
        optimizer=optim.Adam,
        learning_rate=config.training.learning_rate,
    )

    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        gpus=1,
        log_every_n_steps=config.training.log_every_n_steps,
        callbacks=ckpt_callback,
    )

    trainer.fit(
        classifier, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val
    )

    results = trainer.test(classifier, test_dataloaders=dataloader_test)

    print(results)

    extract_logs(classifier.logger.log_dir, classifier.logger.log_dir)

    print(f"Training logs are successfully exported at {classifier.logger.log_dir}")


if __name__ == "__main__":
    config = OmegaConf.load("config.yaml")
    config.merge_with_cli()
    print("hparams")
    print(OmegaConf.to_yaml(config))
    main(config)
