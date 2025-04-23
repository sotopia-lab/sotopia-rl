import os
import torch
import wandb
from datasets import load_dataset
from torch.utils.data import random_split
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)
from accelerate import PartialState
from peft import PeftModelForCausalLM, PeftModelForSequenceClassification
from jinja2 import Environment, FileSystemLoader
from trl import get_kbit_device_map, GRPOConfig, GRPOTrainer
from accelerate import Accelerator
from sotopia_rl.data import GRPODataset
from functools import partial
from typing import List

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"


class SotopiaGRPOTrainer:
    def __init__(self, args, accelerator: Accelerator):
        self.args = args
        self.accelerator = accelerator

        if accelerator.is_main_process:
            self._init_wandb()
        self._setup_tokenizer()
        self._setup_dataset()
        self._create_quantization_config()
        self._setup_policy_models()
        self._setup_classification_models()

        self._setup_grpo_trainer()

        def save_model(self, output_dir: str, _internal_call: bool = False):
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            print(f"Saved PEFT model to {output_dir}")

        self.grpo_trainer.save_model = save_model.__get__(
            self.grpo_trainer, type(self.grpo_trainer)
        )

    def _init_wandb(self):
        wandb.init(
            project=self.args.wandb_project,
            name=self.args.wandb_run_name,
            config={
                k: v
                for k, v in vars(self.args).items()
                if isinstance(v, (int, float, str))
            },
        )

    def _setup_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_name, padding_side="left"
        )
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids("[PAD]")

    def _setup_dataset(self):
        env = Environment(
            loader=FileSystemLoader("/".join(self.args.template_path.split("/")[:-1]))
        )
        template = env.get_template(self.args.template_path.split("/")[-1])
        dataset = GRPODataset(
            data_path=self.args.grpo_data_path,
            tokenizer=self.tokenizer,
            template=template,
            max_length=self.args.max_length,
        )

        generator = torch.Generator().manual_seed(42)
        val_ratio = getattr(self.args, "val_ratio", 0.05)
        train_size = min(int(len(dataset) * (1 - val_ratio)), len(dataset) - 2)
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(
            dataset, [train_size, val_size], generator=generator
        )
        print(
            f"Dataset split: {len(self.train_dataset)} train, {len(self.val_dataset)} validation"
        )

    def _create_quantization_config(self):
        self.quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    def _setup_policy_models(self):
        if self.args.use_lora_train_grpo:
            base_gen_policy = AutoModelForCausalLM.from_pretrained(
                self.args.model_name,
                torch_dtype="auto",
                quantization_config=self.quant_config,
                device_map=get_kbit_device_map(),
            )
            self.policy = PeftModelForCausalLM.from_pretrained(
                base_gen_policy,
                self.args.policy_adapter_path,
                is_trainable=True,
                adapter_name="policy_adapter",
            )
        else:
            self.policy = AutoModelForCausalLM.from_pretrained(
                self.args.model_name,
                torch_dtype="auto",
            )
        self.policy.config.pad_token_id = self.tokenizer.pad_token_id

        requires_grad_num = 0
        for name, param in self.policy.named_parameters():
            print(name, param.requires_grad)
            if param.requires_grad:
                requires_grad_num += 1
        print(f"Number of trainable parameters in policy: {requires_grad_num}")

    def _setup_classification_models(self):

        if self.args.use_lora_train_grpo:
            print("using lora for reward model")
            base_reward_model = AutoModelForSequenceClassification.from_pretrained(
                self.args.model_name,
                torch_dtype="auto",
                num_labels=1,
                quantization_config=self.quant_config,
                device_map=get_kbit_device_map(),
            )
            self.reward_model = PeftModelForSequenceClassification.from_pretrained(
                base_reward_model,
                self.args.reward_adapter_path,
                is_trainable=False,
                adapter_name="value_adapter",
            )
        else:
            self.reward_model = AutoModelForSequenceClassification.from_pretrained(
                self.args.model_name,
                torch_dtype="auto",
                num_labels=1,
            )
        self.reward_model.config.pad_token_id = self.tokenizer.pad_token_id

        def wrapped_reward(
            prompts: list[str], completions: list[str], **kwargs
        ) -> list[float]:
            eos = self.tokenizer.eos_token
            completions = [c + eos for c in completions]

            texts = [p + c for p, c in zip(prompts, completions)]

            inputs = self.tokenizer(
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,
            ).to(self.accelerator.device)

            with torch.inference_mode():
                logits = self.reward_model(**inputs).logits[:, 0]
            return logits.cpu().tolist()

        self.wrapped_reward = wrapped_reward
        for p in self.reward_model.parameters():
            p.requires_grad = False
        requires_grad_num = 0
        for name, param in self.reward_model.named_parameters():
            print(name, param.requires_grad)
            if param.requires_grad:
                requires_grad_num += 1
        print(f"Number of trainable parameters in reward: {requires_grad_num}")

    def _setup_grpo_trainer(self):
        num_processes = self.accelerator.num_processes
        global_batch_size = (
            self.args.per_device_train_batch_size
            * num_processes
            * self.args.gradient_accumulation_steps
        )

        print(
            f"Using num_generations = {self.args.num_generations} (global_batch_size = {global_batch_size})"
        )

        training_args = GRPOConfig(
            disable_dropout=True,
            max_prompt_length=4096,
            logging_steps=1,
            report_to="wandb",
            per_device_train_batch_size=self.args.per_device_train_batch_size,
            per_device_eval_batch_size=self.args.per_device_eval_batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            num_train_epochs=self.args.num_train_epochs,
            learning_rate=self.args.learning_rate,
            output_dir=self.args.output_dir,
            save_steps=self.args.save_steps,
            num_generations=self.args.num_generations,
            log_completions=True,
            wandb_log_unique_prompts=True,
            beta=1e-4,
        )

        self.grpo_trainer = GRPOTrainer(
            args=training_args,
            model=self.policy,
            reward_funcs=self.wrapped_reward,
            processing_class=self.tokenizer,
            reward_processing_classes=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
        )
        print("GRPOtrainer setup complete")

    def train(self):
        try:
            print("Starting GRPO training...")
            train_stats = self.grpo_trainer.train()
            if self.accelerator.is_main_process:
                print("Saving final model checkpoint...")
                self.grpo_trainer.save_model(self.args.output_dir)
            return train_stats
        except Exception as e:
            print(f"Training error: {str(e)}")
            raise