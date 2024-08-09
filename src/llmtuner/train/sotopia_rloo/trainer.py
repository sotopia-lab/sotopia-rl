import math
import os
import sys
from types import MethodType
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import GenerationConfig, Trainer, TrainerControl, TrainerState
from transformers.optimization import get_scheduler
from transformers.trainer_pt_utils import remove_dummy_checkpoint
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME
from trl.trainer.rloo_config import RLOOConfig
from trl.trainer.rloo_trainer import RLOOTrainer
from trl.core import logprobs_from_logits

from ...extras.callbacks import FixValueHeadModelCallback, LogCallback
from ...extras.logging import get_logger
from ...extras.misc import (
    AverageMeter,
    count_parameters,
    get_current_device,
    get_logits_processor,
)
from ..utils import create_custom_optimzer, create_custom_scheduler
from .utils import (
    dump_layernorm,
    get_rewards_from_server,
    replace_model,
    restore_layernorm,
)

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import (
        DataCollatorWithPadding,
        PreTrainedTokenizer,
        Seq2SeqTrainingArguments,
        TrainerCallback,
    )
    from trl import AutoModelForCausalLMWithValueHead

    from ...hparams import FinetuningArguments, GeneratingArguments, ModelArguments


logger = get_logger(__name__)


class CustomRLOOTrainer(RLOOTrainer, Trainer):
    r"""
    Inherits RLOOTrainer.
    """

    def __init__(
        self,
        model_args: "ModelArguments",
        training_args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
        generating_args: "GeneratingArguments",
        callbacks: List["TrainerCallback"],
        model: "AutoModelForCausalLMWithValueHead",
        reward_model: Optional["AutoModelForCausalLMWithValueHead"],
        ref_model: Optional["AutoModelForCausalLMWithValueHead"],
        tokenizer: "PreTrainedTokenizer",
        dataset: "Dataset",
        data_collator: "DataCollatorWithPadding",
    ):
        backward_batch_size = (
            training_args.per_device_train_batch_size
            * training_args.gradient_accumulation_steps
        )
        rloo_config = RLOOConfig(
            sft_model_path=model_args.model_name_or_path,
            learning_rate=training_args.learning_rate,
            mini_batch_size=training_args.per_device_train_batch_size,
            batch_size=backward_batch_size * finetuning_args.ppo_buffer_size,
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
            max_grad_norm=training_args.max_grad_norm,
            seed=training_args.seed,
            whiten_rewards=finetuning_args.ppo_whiten_rewards,
            output_dir=training_args.output_dir,
        )


        # Create optimizer and scheduler
        if training_args.max_steps > 0:
            num_training_steps = training_args.max_steps
        else:
            total_train_batch_size = (
                backward_batch_size
                * finetuning_args.ppo_buffer_size
                * training_args.world_size
            )
            num_training_steps = training_args.num_train_epochs * math.ceil(
                len(dataset) / total_train_batch_size
            )

        
        model.generation_config = GenerationConfig(
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=[tokenizer.eos_token_id]
            + tokenizer.additional_special_tokens_ids,
            **generating_args.to_dict(),
        )

        train_dataset = []
        eval_dataset = []
        train_len = int(len(dataset) * 0.8)
        for i, data in enumerate(dataset):
            if i < train_len:
                train_dataset.append({'input_ids': data['input_ids'], 'attention_mask': data['attention_mask']})
            else:
                eval_dataset.append({'input_ids': data['input_ids'], 'attention_mask': data['attention_mask']})

        RLOOTrainer.__init__(
            self,
            config=rloo_config,
            tokenizer=tokenizer,
            policy=model,
            ref_policy=ref_model,
            reward_model=reward_model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        self.model_args = model_args
        self.finetuning_args = finetuning_args
        self.reward_model = reward_model
        self.current_device = (
            get_current_device()
        )  # patch for deepspeed training

        self.generation_config = GenerationConfig(
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=[self.tokenizer.eos_token_id]
            + self.tokenizer.additional_special_tokens_ids,
            **generating_args.to_dict(),
        )

        self.state = TrainerState()
        self.control = TrainerControl()
        self.is_deepspeed_enabled = (
            self.accelerator.distributed_type == "DEEPSPEED"
            and hasattr(self.accelerator.state, "deepspeed_plugin")
        )
        self.log_callback, self.save_callback = callbacks[0], callbacks[1]
        assert isinstance(self.log_callback, LogCallback) and isinstance(
            self.save_callback, FixValueHeadModelCallback
        )

        if self.args.max_steps > 0:
            logger.info(
                "max_steps is given, it will override any value given in num_train_epochs"
            )

        print(finetuning_args)
        if finetuning_args.reward_model_type == "full":
            if self.is_deepspeed_enabled:
                if not (
                    getattr(
                        reward_model.pretrained_model,
                        "is_loaded_in_8bit",
                        False,
                    )
                    or getattr(
                        reward_model.pretrained_model,
                        "is_loaded_in_4bit",
                        False,
                    )
                ):  # quantized models are already set on the correct device
                    self.reward_model = self._prepare_deepspeed(
                        self.reward_model
                    )
            else:
                self.reward_model = self.accelerator.prepare_model(
                    self.reward_model, evaluation_mode=True
                )

        if finetuning_args.use_badam:
            from badam import clip_grad_norm_for_sparse_tensor

            self.accelerator.clip_grad_norm_ = MethodType(
                clip_grad_norm_for_sparse_tensor, self.accelerator
            )

    def rloo_train(self, resume_from_checkpoint: Optional[str] = None) -> None:
        self.train()

    def save_model(self, output_dir: Optional[str] = None) -> None:
        r"""
        Saves model checkpoint.

        Subclass and override to inject custom behavior.
        """
        if self.args.should_save:
            try:
                self._save(
                    output_dir,
                    state_dict=self.accelerator.get_state_dict(self.model),
                )
            except ValueError:
                logger.warning(
                    " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead,"
                    " use zero_to_fp32.py to recover weights"
                )
                self._save(output_dir, state_dict={})
                remove_dummy_checkpoint(
                    True, output_dir, [WEIGHTS_NAME, SAFE_WEIGHTS_NAME]
                )
                self.model.save_checkpoint(output_dir)
