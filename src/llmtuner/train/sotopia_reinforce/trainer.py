import json
import math
import os
import sys
from types import MethodType
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import torch
from tqdm import tqdm
from transformers import Seq2SeqTrainer

from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ...extras.misc import AverageMeter
from ..utils import create_custom_optimzer, create_custom_scheduler

if TYPE_CHECKING:
    from transformers.trainer import PredictionOutput
    from trl import AutoModelForCausalLMWithValueHead

    from ...hparams import FinetuningArguments


logger = get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""
    Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE.
    """

    def __init__(
        self, 
        finetuning_args: "FinetuningArguments", 
        reward_model: Optional["AutoModelForCausalLMWithValueHead"],
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args
        if finetuning_args.use_badam:
            from badam import clip_grad_norm_for_sparse_tensor

            self.accelerator.clip_grad_norm_ = MethodType(
                clip_grad_norm_for_sparse_tensor, self.accelerator
            )
        self.reward_model = reward_model


    def compute_loss(self, model, inputs, return_outputs=False):
        generated_ids = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=200,
            early_stopping=True,
            num_beams=5,
            num_return_sequences=5,
        )

        # Decode the generated sequences
        candidates = [self.tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]

        reward_model_inputs_list = self.prepare_rm_inputs(inputs, candidates)
        for reward_model_inputs in reward_model_inputs_list:
            _, _, value = self.reward_model(**reward_model_inputs)

        # Custom loss computation
        outputs = model(**inputs)
        import pdb; pdb.set_trace()
        logits = outputs.get("logits")
        labels = inputs.get("labels")
        loss = self.custom_loss_function(logits, labels)
        return (loss, outputs) if return_outputs else loss
     

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

    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[
        Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]
    ]:
        r"""
        Removes the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        labels = (
            inputs["labels"].detach().clone() if "labels" in inputs else None
        )  # backup labels
        if self.args.predict_with_generate:
            assert (
                self.tokenizer.padding_side == "left"
            ), "This method only accepts left-padded tensor."
            prompt_len, label_len = inputs["input_ids"].size(-1), inputs[
                "labels"
            ].size(-1)
            if prompt_len > label_len:
                inputs["labels"] = self._pad_tensors_to_target_len(
                    inputs["labels"], inputs["input_ids"]
                )
            if (
                label_len > prompt_len
            ):  # truncate the labels instead of padding the inputs (llama2 fp16 compatibility)
                inputs["labels"] = inputs["labels"][:, :prompt_len]

        (
            loss,
            generated_tokens,
            _,
        ) = super().prediction_step(  # ignore the returned labels (may be truncated)
            model,
            inputs,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, :prompt_len] = self.tokenizer.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def _pad_tensors_to_target_len(
        self, src_tensor: torch.Tensor, tgt_tensor: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Pads the tensor to the same length as the target tensor.
        """
        assert (
            self.tokenizer.pad_token_id is not None
        ), "Pad token is required."
        padded_tensor = self.tokenizer.pad_token_id * torch.ones_like(
            tgt_tensor
        )
        padded_tensor[
            :, -src_tensor.shape[-1] :
        ] = src_tensor  # adopt left-padding
        return padded_tensor.contiguous()  # in contiguous memory

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

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX,
            predict_results.label_ids,
            self.tokenizer.pad_token_id,
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.tokenizer.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.tokenizer.pad_token_id)[0]
            if len(pad_len):
                preds[i] = np.concatenate(
                    (preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1
                )  # move pad token to last

        decoded_labels = self.tokenizer.batch_decode(
            labels,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        decoded_preds = self.tokenizer.batch_decode(
            preds, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: List[str] = []
            for label, pred in zip(decoded_labels, decoded_preds):
                res.append(
                    json.dumps(
                        {"label": label, "predict": pred}, ensure_ascii=False
                    )
                )
            writer.write("\n".join(res))

    def prepare_rm_inputs(
        self, inputs: Dict[str, torch.Tensor], candidates: List[str]
    ) -> Dict[str, torch.Tensor]:
        r"""
        Prepares inputs for the reward model.
        """
        inputs = self.tokenizer(
            candidates,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return inputs