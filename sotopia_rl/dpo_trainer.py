import json
import os

import torch
import wandb
from accelerate import Accelerator
from datasets import Dataset
from jinja2 import Environment, FileSystemLoader
from peft import LoraConfig, PeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import DPOConfig, DPOTrainer, get_kbit_device_map

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class SotopiaDPOTrainer:
    """DPO Trainer for Sotopia using preference pairs (chosen/rejected responses)."""

    def __init__(self, args, accelerator: Accelerator):
        self.args = args
        self.accelerator = accelerator

        if accelerator.is_main_process:
            self._init_wandb()
        self._setup_tokenizer()
        self._setup_dataset()
        self._create_quantization_config()
        self._setup_models()
        self._setup_dpo_trainer()

        # Custom save_model to save PEFT adapter
        def save_model(self, output_dir: str, _internal_call: bool = False):
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            print(f"Saved PEFT model to {output_dir}")

        self.dpo_trainer.save_model = save_model.__get__(
            self.dpo_trainer, type(self.dpo_trainer)
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
        """Setup dataset using HuggingFace datasets for TRL compatibility."""
        # Load template for prompt rendering
        env = Environment(
            loader=FileSystemLoader("/".join(self.args.template_path.split("/")[:-1]))
        )
        template = env.get_template(self.args.template_path.split("/")[-1])

        # Load raw data
        with open(self.args.dpo_data_path, "r") as f:
            raw_data = json.load(f)

        # Process data into DPO format with rendered prompts
        processed_data = []
        for item in raw_data:
            rendered_prompt = template.render(
                messages=[{"role": "user", "content": item["input"]}],
                add_generation_prompt=True
            )
            processed_data.append({
                "prompt": rendered_prompt,
                "chosen": item["chosen"],
                "rejected": item["rejected"],
            })

        # Create HuggingFace Dataset
        full_dataset = Dataset.from_list(processed_data)

        # Split into train/val
        val_ratio = getattr(self.args, "val_ratio", 0.05)
        split_dataset = full_dataset.train_test_split(test_size=val_ratio, seed=42)
        self.train_dataset = split_dataset["train"]
        self.val_dataset = split_dataset["test"]

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

    def _setup_models(self):
        """Setup policy model and reference model for DPO."""
        if self.args.use_lora:
            # Load base model with quantization for LoRA training
            base_model = AutoModelForCausalLM.from_pretrained(
                self.args.model_name,
                torch_dtype="auto",
                quantization_config=self.quant_config,
                device_map=get_kbit_device_map(),
            )

            # Configure LoRA
            peft_config = LoraConfig(
                r=self.args.lora_r,
                lora_alpha=self.args.lora_alpha,
                lora_dropout=self.args.lora_dropout,
                target_modules=self.args.target_modules.split(","),
                bias="none",
                task_type="CAUSAL_LM",
            )

            # If we have a pre-trained adapter, load it
            if hasattr(self.args, "policy_adapter_path") and self.args.policy_adapter_path:
                self.model = PeftModelForCausalLM.from_pretrained(
                    base_model,
                    self.args.policy_adapter_path,
                    is_trainable=True,
                    adapter_name="policy_adapter",
                )
            else:
                # Create new PEFT model with LoRA config
                from peft import get_peft_model
                self.model = get_peft_model(base_model, peft_config)
            
            self.peft_config = peft_config
        else:
            # Full fine-tuning without LoRA
            self.model = AutoModelForCausalLM.from_pretrained(
                self.args.model_name,
                torch_dtype="auto",
            )
            self.peft_config = None

        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # Setup reference model (frozen)
        if hasattr(self.args, "ref_adapter_path") and self.args.ref_adapter_path:
            # Load reference model with adapter
            base_ref = AutoModelForCausalLM.from_pretrained(
                self.args.model_name,
                torch_dtype="auto",
                quantization_config=self.quant_config,
                device_map=get_kbit_device_map(),
            )
            self.ref_model = PeftModelForCausalLM.from_pretrained(
                base_ref,
                self.args.ref_adapter_path,
                is_trainable=False,
                adapter_name="ref_adapter",
            )
        else:
            # Use None to let DPOTrainer create reference model automatically
            self.ref_model = None

        # Log trainable parameters
        requires_grad_num = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                requires_grad_num += 1
        print(f"Number of trainable parameters in model: {requires_grad_num}")

    def _setup_dpo_trainer(self):
        num_processes = self.accelerator.num_processes
        global_batch_size = (
            self.args.per_device_train_batch_size
            * num_processes
            * self.args.gradient_accumulation_steps
        )

        print(f"Global batch size = {global_batch_size}")

        training_args = DPOConfig(
            output_dir=self.args.output_dir,
            per_device_train_batch_size=self.args.per_device_train_batch_size,
            per_device_eval_batch_size=self.args.per_device_eval_batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            num_train_epochs=self.args.num_train_epochs,
            learning_rate=self.args.learning_rate,
            beta=self.args.beta,  # DPO temperature parameter
            max_length=self.args.max_length,
            max_prompt_length=getattr(self.args, "max_prompt_length", 2048),
            logging_steps=1,
            save_steps=self.args.save_steps,
            report_to="wandb",
            bf16=True,
            remove_unused_columns=False,
            ddp_find_unused_parameters=False,
            eval_strategy="steps",
            eval_steps=self.args.save_steps,
        )

        self.dpo_trainer = DPOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            processing_class=self.tokenizer,
            peft_config=self.peft_config if not hasattr(self.args, "policy_adapter_path") or not self.args.policy_adapter_path else None,
        )
        print("DPOTrainer setup complete")

    def train(self):
        try:
            print("Starting DPO training...")
            train_stats = self.dpo_trainer.train()
            if self.accelerator.is_main_process:
                print("Saving final model checkpoint...")
                self.dpo_trainer.save_model(self.args.output_dir)
            return train_stats
        except Exception as e:
            print(f"Training error: {str(e)}")
            raise

