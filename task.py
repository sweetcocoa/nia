import torch

from typing import Callable, Mapping, Optional, Sequence, Union
from flash.core.classification import ClassificationTask
import torchmetrics
from omegaconf import OmegaConf


class ClassificationTaskHpSave(ClassificationTask):
    def __init__(
        self,
        model,
        config,
        num_classes: Optional[int] = None,
        loss_fn: Optional[Callable] = None,
        metrics: Union[torchmetrics.Metric, Mapping, Sequence, None] = None,
        multi_label: bool = False,
        **kwargs
    ) -> None:

        self.lr = config.training.learning_rate
        kwargs["learning_rate"] = self.lr

        super().__init__(
            model,
            num_classes=num_classes,
            loss_fn=loss_fn,
            metrics=metrics,
            multi_label=multi_label,
            **kwargs
        )
        self.metrics = torchmetrics.F1(num_classes)

        self.save_hyperparameters(OmegaConf.to_container(config))
        self.save_hyperparameters("loss_fn", "metrics", "num_classes")

    def on_fit_end(self) -> None:
        trainer = self.trainer
        trainer.logger.log_metrics(
            {
                "val_f1": trainer.callback_metrics["val_f1"],
            },
            step=trainer.global_step,
        )

        return super().on_fit_end()
