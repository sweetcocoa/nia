import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from omegaconf import OmegaConf
from task import ClassificationTaskHpSave

# from flash.core.classification import ClassificationTask

from model import SimpleAudioClassificationModel
from training_utils import extract_logs, get_dataset


def main(config):

    dataset_train = get_dataset(
        config.dataset.path,
        phase="train",
        pipe_config=config.pipe,
        use_major_class=config.dataset.use_major_class,
    )
    dataset_val = get_dataset(
        config.dataset.path,
        phase="val",
        pipe_config=config.pipe,
        use_major_class=config.dataset.use_major_class,
    )
    dataset_test = get_dataset(
        config.dataset.path,
        phase="test",
        pipe_config=config.pipe,
        use_major_class=config.dataset.use_major_class,
    )

    print(
        f"""
Dataset Summary:
# of train : {len(dataset_train)}, ({len(dataset_train.classes)})
# of val : {len(dataset_val)}, ({len(dataset_val.classes)})
# of test : {len(dataset_test)}, ({len(dataset_test.classes)})
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

    model = SimpleAudioClassificationModel(
        num_classes=len(dataset_val.classes),
        model=config.model,
    )

    # multi_label=True, num_class 가 specify됐을 때 f1 metric이 사용됨.
    classifier = ClassificationTaskHpSave(
        model,
        config,
        loss_fn=nn.functional.cross_entropy,
        num_classes=len(dataset_val.classes),
        multi_label=True,
        optimizer=optim.Adam,
        learning_rate=config.training.learning_rate,
    )

    logger = TensorBoardLogger(config.training.logdir, version=config.training.version)

    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        gpus=1,
        logger=logger,
        log_every_n_steps=config.training.log_every_n_steps,
        callbacks=ckpt_callback,
    )

    os.makedirs(trainer.log_dir, exist_ok=True)
    torch.save(
        dataset_train.class_to_idx,
        os.path.join(trainer.log_dir, "class_to_idx.pth"),
    )
    torch.save(
        dataset_train.idx_to_class,
        os.path.join(trainer.log_dir, "idx_to_class.pth"),
    )
    OmegaConf.save(config, os.path.join(trainer.log_dir, "config.yaml"))

    trainer.fit(
        classifier, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val
    )

    results = trainer.test(test_dataloaders=dataloader_test, ckpt_path="best")
    trainer.logger.log_metrics(
        {"hp_metric": trainer.callback_metrics["test_f1"]}, step=trainer.global_step
    )

    print(results)

    extract_logs(classifier.logger.log_dir, classifier.logger.log_dir)

    print(f"Training logs are successfully exported at {classifier.logger.log_dir}")


if __name__ == "__main__":
    config = OmegaConf.load("config.yaml")
    config.merge_with_cli()
    print("hparams")
    print(OmegaConf.to_yaml(config))
    main(config)
