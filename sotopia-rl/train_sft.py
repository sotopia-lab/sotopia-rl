import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments
from trl import AutoModelForCausalLMWithValueHead, SFTTrainer
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, random_split
from data import SFTDataset
from jinja2 import Environment, FileSystemLoader
import wandb
import os
import argparse

from transformers import AutoModelForCausalLM


class SFTTrainerWrapper:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize wandb
        wandb.init(
            project=args.wandb_project, 
            name=args.wandb_run_name,
            config={k: v for k, v in vars(args).items() if isinstance(v, (int, float, str))}
        )

        # Load the tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        self.tokenizer.model_max_length = args.max_length

        # Initialize model with LoRA if specified
        base_model = AutoModelForCausalLM.from_pretrained(args.model_name)
        if args.use_lora:
            peft_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=args.target_modules.split(",")
            )
            self.model = get_peft_model(base_model, peft_config).to(self.device)
        else:
            self.model = base_model.to(self.device)

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
            report_to="wandb",  # Log to wandb
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
        dataset = SFTDataset(self.args.reward_data_path, self.tokenizer, max_length=self.args.max_length, template=self.template)
        train_size = int(0.95 * len(dataset))
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])

    def train(self):
        # Begin training with SFTTrainer
        self.trainer.train()
        # Save the final model
        self.save_lora_checkpoint()

    def save_lora_checkpoint(self):
        # Save model in standard PEFT format if LoRA is used
        if self.args.use_lora:
            checkpoint_path = os.path.join(self.args.output_dir, "best_lora_checkpoint")
            os.makedirs(checkpoint_path, exist_ok=True)
            self.model.save_pretrained(checkpoint_path)
            print(f"LoRA checkpoint saved at {checkpoint_path}")

    def load_lora_checkpoint(self, checkpoint_path):
        # Load LoRA checkpoint in standard PEFT format
        self.model = self.model.from_pretrained(self.model, checkpoint_path)
        print(f"LoRA checkpoint loaded from {checkpoint_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a reward model using SFT with LoRA.")
    parser.add_argument("--local-rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Model name or path")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--train_batch_size", type=int, default=2, help="Training batch size")
    parser.add_argument("--val_batch_size", type=int, default=2, help="Validation batch size")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--reward_data_path", type=str, required=True, help="Path to reward data")
    parser.add_argument("--template_path", type=str, required=True, help="Path to the Jinja template file")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--evaluation_steps", type=int, default=100, help="Evaluation interval in steps")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Gradient accumulation steps")

    # LoRA-specific arguments
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA for fine-tuning")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--target_modules", type=str, default="c_attn,q_proj,v_proj", help="Target modules for LoRA")

    # Checkpoint and Wandb arguments
    parser.add_argument("--checkpoint_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--lora_checkpoint_path", type=str, default=None, help="Path to load LoRA checkpoint")
    parser.add_argument("--wandb_project", type=str, default="sft-project", help="Wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default="sft-run", help="Wandb run name")

    args = parser.parse_args()
    trainer = SFTTrainerWrapper(args)
    trainer.train()
