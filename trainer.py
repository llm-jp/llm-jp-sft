import logging

from transformers.trainer import (
    get_parameter_names,
    ALL_LAYERNORM_LAYERS,
    OptimizerNames,
)
from trl import SFTTrainer

logger = logging.getLogger(__name__)


class CustomSFTTrainer(SFTTrainer):
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method (or `create_optimizer` and/or
        `create_scheduler`) in a subclass.
        """
        self.create_optimizer()
        self.create_scheduler(
            num_training_steps=num_training_steps, optimizer=self.optimizer
        )

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_kwargs = {
                "lr": self.args.learning_rate,
                "betas": (self.args.adam_beta1, self.args.adam_beta2),
            }
            if self.args.optim == OptimizerNames.ADAMW_TORCH:
                from torch.optim import AdamW

                optimizer_cls = AdamW
                optimizer_kwargs["eps"] = self.args.adam_epsilon
            elif self.args.optim == OptimizerNames.LION:
                from lion import Lion

                optimizer_cls = Lion

            self.optimizer = optimizer_cls(
                optimizer_grouped_parameters, **optimizer_kwargs
            )

        return self.optimizer
