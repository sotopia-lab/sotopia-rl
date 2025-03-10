import os

import torch
from jinja2 import Environment, FileSystemLoader
from peft import LoraConfig, get_peft_model
from torch.utils.data import random_split
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

import wandb
from sotopia_rl.data import SFTDataset


class SotopiaSFTTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={k: v for k, v in vars(args).items() if isinstance(v, (int, float, str))}
        )
        config = AutoConfig.from_pretrained(args.model_name)
        config.use_cache = False

        # Load the tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        self.tokenizer.model_max_length = args.max_length

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
        )

        if args.use_qlora:
            print(f"Using QLoRA (4bit) to load model: {args.model_name}")
            base_model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                quantization_config=quantization_config,
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(args.model_name).to(self.device)

        if args.use_lora:
            peft_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=args.target_modules.split(",")
            )
            self.model = get_peft_model(base_model, peft_config)
            if not args.use_qlora:
                self.model = self.model.to(self.device)
        else:
            self.model = base_model

        # Load LoRA checkpoint if specified
        if args.lora_checkpoint_path:
            self.load_lora_checkpoint(args.lora_checkpoint_path)

        env = Environment(loader=FileSystemLoader("/".join(self.args.template_path.split("/")[:-1])))
        self.template = env.get_template(self.args.template_path.split("/")[-1])

        # Set up dataset and data loaders
        self.setup_dataloaders()

        # Set up the SFT Trainer with appropriate arguments
        training_args = TrainingArguments(
            output_dir=args.checkpoint_dir,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.val_batch_size,
            gradient_accumulation_steps=args.accumulation_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            evaluation_strategy="steps",
            eval_steps=args.evaluation_steps,
            save_steps=args.evaluation_steps,
            logging_dir="./logs",
            report_to="wandb",
            gradient_checkpointing=False,
            optim="paged_adamw_8bit" if args.use_qlora else "adamw_torch",
            fp16=True,
        )

        # Initialize SFTTrainer
        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.collate_fn,
        )

    def collate_fn(self, batch):
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [item["input_ids"] for item in batch], batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_masks = torch.nn.utils.rnn.pad_sequence(
            [item["attention_mask"] for item in batch], batch_first=True, padding_value=0
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [item["labels"] for item in batch], batch_first=True, padding_value=-100
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "labels": labels,
        }

    def setup_dataloaders(self):
        # Load dataset and create train/val split
        dataset = SFTDataset(self.args.sft_data_path, self.tokenizer, max_length=self.args.max_length,
                             template=self.template)
        train_size = int(0.95 * len(dataset))
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])
        print(f"Training dataset size: {len(self.train_dataset)}")
        print(f"Validation dataset size: {len(self.val_dataset)}")

    def train(self):
        # Begin training with SFTTrainer
        self.trainer.train()

        # Evaluate one last time to log final metrics
        eval_results = self.trainer.evaluate()
        wandb.log({"final_eval_loss": eval_results["eval_loss"]})

        # Save the final model
        self.save_lora_checkpoint()

    def save_lora_checkpoint(self):
        # Save model in standard PEFT format if LoRA is used
        if self.args.use_lora:
            checkpoint_path = os.path.join(self.args.checkpoint_dir, "best_lora_checkpoint")
            os.makedirs(checkpoint_path, exist_ok=True)
            self.model.save_pretrained(checkpoint_path)
            print(f"LoRA checkpoint saved at {checkpoint_path}")

    def load_lora_checkpoint(self, checkpoint_path):
        # Load LoRA checkpoint in standard PEFT format
        self.model = self.model.from_pretrained(self.model, checkpoint_path)
        print(f"LoRA checkpoint loaded from {checkpoint_path}")
