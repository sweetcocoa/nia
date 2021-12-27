import os
import importlib
import math
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from omegaconf import OmegaConf

from task import ClassificationTaskHpSave
from model import SimpleAudioClassificationModel
from training_utils import extract_logs, get_dataset
from inference_utils import Inference
from loss.cross_entropy import CrossEntropyWithLabelSmoothing


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

    config.model.num_classes = len(dataset_val.classes)

    weighted_sampler = None
    if config.training.weighted_sampling:
        weighted_sampler = dataset_train.get_weighted_sampler()

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=config.training.batch_size,
        sampler=weighted_sampler,
        shuffle=True if weighted_sampler is None else False,
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
        save_top_k=3,
        mode="max",
        save_last=True,
    )

    model = SimpleAudioClassificationModel(
        model_config=config.model,
        pipe_config=config.pipe,
    )

    optimizer = importlib.import_module(
        "optimizer." + config.training.optimizer
    ).__getattribute__("get_optimizer")

    if config.training.lr_scheduler:
        get_scheduler = importlib.import_module(
            "scheduler." + config.training.lr_scheduler
        ).__getattribute__("get_scheduler")
        scheduler, scheduler_kwargs = get_scheduler()
    else:
        scheduler, scheduler_kwargs = None, None

    loss_fn = CrossEntropyWithLabelSmoothing(smoothing=config.training.label_smoothing)

    # multi_label=True, num_class 가 specify됐을 때 f1 metric이 사용됨.
    classifier = ClassificationTaskHpSave(
        model,
        config,
        loss_fn=loss_fn,
        num_classes=len(dataset_val.classes),
        multi_label=True,
        optimizer=optimizer,
        scheduler=scheduler,
        scheduler_kwargs=scheduler_kwargs,
        learning_rate=config.training.learning_rate,
    )

    logger = TensorBoardLogger(config.training.logdir, version=config.training.version)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        default_root_dir=logger.log_dir,
        gpus=1,
        logger=logger,
        log_every_n_steps=config.training.log_every_n_steps,
        callbacks=[ckpt_callback, lr_monitor],
        auto_lr_find=config.training.auto_lr_find,
    )

    if config.training.auto_lr_find:
        trainer.tune(
            classifier,
            train_dataloaders=dataloader_train,
            val_dataloaders=dataloader_val,
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
    dataset_train.save_counter(os.path.join(trainer.log_dir, "stat_train.txt"))
    dataset_val.save_counter(os.path.join(trainer.log_dir, "stat_val.txt"))
    dataset_test.save_counter(os.path.join(trainer.log_dir, "stat_test.txt"))

    trainer.fit(
        classifier, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val
    )

    print("test f1 will be measured on :", trainer.checkpoint_callback.best_model_path)
    results = trainer.test(test_dataloaders=dataloader_test, ckpt_path="best")
    trainer.logger.log_metrics(
        {"hp_metric": trainer.callback_metrics["test_f1"]}, step=trainer.global_step
    )

    print(results)

    extract_logs(classifier.logger.log_dir, classifier.logger.log_dir)

    print(f"Training logs are successfully exported at {classifier.logger.log_dir}")

    best_model = trainer.checkpoint_callback.best_model_path
    inference = Inference(best_model, config, "cuda", model=model)
    output_dir = classifier.logger.log_dir
    cm = inference.get_confusion_matrix_of_dataset(
        config.dataset.path, config, output_dir=output_dir
    )
    print(f"Confusion matrix Images can be seen at checkpoint's folder {output_dir})")
    print(cm)


if __name__ == "__main__":
    config = OmegaConf.load("config.yaml")
    config.merge_with_cli()
    print("hparams")
    print(OmegaConf.to_yaml(config))
    main(config)
