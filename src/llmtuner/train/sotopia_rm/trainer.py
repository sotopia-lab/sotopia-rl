import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch
from transformers import Trainer

from ...extras.logging import get_logger
from ..utils import create_custom_optimzer, create_custom_scheduler

if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = get_logger(__name__)


class SotopiaRMTrainer(Trainer):
    r"""
    Inherits Trainer to compute mse loss.
    """

    def __init__(
        self, finetuning_args: "FinetuningArguments", **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args
        self.can_return_loss = True  # override property to return eval_loss
        if finetuning_args.use_badam:
            from badam import clip_grad_norm_for_sparse_tensor

            self.accelerator.clip_grad_norm_ = MethodType(
                clip_grad_norm_for_sparse_tensor, self.accelerator
            )

    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimzer(
                self.model, self.args, self.finetuning_args
            )
        return super().create_optimizer()

    def create_scheduler(
        self,
        num_training_steps: int,
        optimizer: Optional["torch.optim.Optimizer"] = None,
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    def compute_loss(
        self,
        model: "PreTrainedModel",
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        # Compute rewards
        batch_size = inputs["input_ids"].size(0)
        reg_labels = inputs.pop("reg_labels", None).view(-1)
        _, _, values = model(
            **inputs, output_hidden_states=True, return_dict=True
        )
        last_indices = (inputs["attention_mask"].sum(dim=1) - 1).long()
        
        # Use these indices to extract the corresponding values from 'values'
        eos_values = values[torch.arange(values.size(0)), last_indices]

        # Compute the MSE loss
        loss = torch.nn.functional.mse_loss(eos_values, reg_labels)
        loss = loss / batch_size
        if return_outputs:
            return loss, [loss, eos_values, eos_values]
        return loss

    def save_predictions(self, predict_results: "PredictionOutput") -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(
            self.args.output_dir, "generated_predictions.jsonl"
        )
        logger.info(f"Saving prediction results to {output_prediction_file}")
        chosen_scores, _ = predict_results.predictions
        # import pdb; pdb.set_trace()

        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: List[str] = []
            for score in chosen_scores:
                res.append(
                    json.dumps(
                        {
                            "score": round(float(score), 2),
                        }
                    )
                )
            writer.write("\n".join(res))
