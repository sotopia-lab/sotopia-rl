import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, random_split
from torch.nn import MSELoss
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from jinja2 import Environment, FileSystemLoader
from data import RMDataset
from tqdm import tqdm
import os
import wandb
import argparse
from transformers import Trainer, TrainingArguments

class RMTrainer(Trainer):
    def __init__(self, args, **kwargs):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_val_loss = float('inf')  # Initialize best validation loss
        train_dataset, eval_dataset = self.setup_dataset()

        # Initialize wandb
        wandb.init(
            project=args.wandb_project, 
            name=args.wandb_run_name,
            config={k: v for k, v in vars(args).items() if isinstance(v, (int, float, str))}
        )

        # Initialize model with LoRA
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules.split(",")
        )
        model = AutoModelForCausalLMWithValueHead.from_pretrained(args.model_name)
        model = get_peft_model(model, peft_config).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)


        # Load LoRA checkpoint if specified
        if args.lora_checkpoint_path:
            self.load_lora_checkpoint(model, args.lora_checkpoint_path)

        # Define optimizer and learning rate scheduler
        optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        lr_scheduler = self.create_lr_scheduler(optimizer, len(train_dataset))

        # Additional training arguments can be added here
        training_args = TrainingArguments(
            output_dir=args.checkpoint_dir,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.val_batch_size,
            num_train_epochs=args.num_epochs,
            evaluation_strategy="steps",
            logging_steps=50,
            save_steps=args.evaluation_steps,
            eval_steps=args.evaluation_steps,
            logging_dir="./logs",
            gradient_accumulation_steps=args.accumulation_steps,
            warmup_steps=int(len(train_dataset) * args.warmup_epochs),
            load_best_model_at_end=True,
            save_total_limit=10,  # Only keep the latest 3 checkpoints
            label_names=["labels"],
        )

        # Initialize Trainer with parent class constructor
        super().__init__(
            model=model,
            args=training_args,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            optimizers=(optimizer, lr_scheduler),
            data_collator=self.collate_fn,
            **kwargs
        )

        self.loss_fn = MSELoss()


    def collate_fn(self, batch):
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [item["input_ids"] for item in batch], batch_first=True, padding_value=self.processing_class.pad_token_id
        )
        attention_masks = torch.nn.utils.rnn.pad_sequence(
            [item["attention_mask"] for item in batch], batch_first=True, padding_value=0
        )
        labels = torch.stack([item["labels"] for item in batch])  # Stack labels into a tensor

        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "labels": labels,  # Ensure labels are present in the batch output
        }

    def setup_dataset(self):
        # Load dataset and create train/val split
        env = Environment(loader=FileSystemLoader("/".join(self.args.template_path.split("/")[:-1])))
        template = env.get_template(self.args.template_path.split("/")[-1])
        tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)
        dataset = RMDataset(self.args.reward_data_path, tokenizer, template, self.args.max_length)
        
        train_size = int(0.99 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        return train_dataset, val_dataset

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Custom compute_loss for value head-based reward model
        input_ids = inputs["input_ids"].to(self.device)
        attention_masks = inputs["attention_mask"].to(self.device)
        true_rewards = inputs["labels"].to(self.device)
        
        # Forward pass
        _, _, outputs = model(input_ids, attention_mask=attention_masks, return_dict=True)
        
        # Compute loss using value head outputs
        last_indices = (attention_masks.sum(dim=1) - 1).long()
        eos_values = outputs[torch.arange(outputs.size(0)), last_indices]
        loss = self.loss_fn(eos_values, true_rewards)
        
        return (loss, outputs) if return_outputs else loss

    def create_lr_scheduler(self, optimizer, steps_per_epoch):
        warmup_scheduler = LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=int(steps_per_epoch * self.args.warmup_epochs)
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=self.args.num_epochs * steps_per_epoch, eta_min=self.args.min_lr
        )
        return SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[int(steps_per_epoch * self.args.warmup_epochs)])

    def save_model(self, output_dir=None, **kwargs):
        # Save the LoRA model checkpoint and tokenizer
        super().save_model(output_dir)  # This will save model, tokenizer, config, etc.
        os.makedirs(output_dir, exist_ok=True)
        
        # Save LoRA-specific components (value head and LoRA parameters)
        torch.save(self.model.v_head.state_dict(), os.path.join(output_dir, "value_head_state_dict.pt"))
        
        # Save LoRA parameters
        peft_params = {
            name: param.data 
            for name, param in self.model.named_parameters() 
            if 'v_head' not in name and 'lora' in name
        }
        torch.save(peft_params, os.path.join(output_dir, "lora_parameters.pt"))
        
        print(f"LoRA checkpoint saved at {output_dir}")

    def load_lora_checkpoint(self, model, checkpoint_path):
        # Load LoRA-specific checkpoints
        value_head_path = os.path.join(checkpoint_path, "value_head_state_dict.pt")
        if os.path.exists(value_head_path):
            model.v_head.load_state_dict(torch.load(value_head_path))
        
        lora_params_path = os.path.join(checkpoint_path, "lora_parameters.pt")
        if os.path.exists(lora_params_path):
            lora_params = torch.load(lora_params_path)
            set_peft_model_state_dict(model, lora_params)

        print(f"LoRA and value head checkpoint loaded from {checkpoint_path}")


    def load_lora_checkpoint(self, checkpoint_path):
        value_head_path = os.path.join(checkpoint_path, "value_head_state_dict.pt")
        if os.path.exists(value_head_path):
            self.model.v_head.load_state_dict(
                torch.load(value_head_path)
            )
        
        lora_params_path = os.path.join(checkpoint_path, "lora_parameters.pt")
        if os.path.exists(lora_params_path):
            lora_params = torch.load(lora_params_path)
            
            model_state_dict = self.model.state_dict()
            for name, param in lora_params.items():
                if name in model_state_dict:
                    model_state_dict[name].copy_(param)
            
        print(f"LoRA and value head checkpoint loaded from {checkpoint_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a reward model with value head using LoRA.")
    # Define arguments as before
    parser.add_argument("--model_name", type=str, default="/data/models/gemma-2-2b-it", help="Path to the model")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for optimizer")
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--val_batch_size", type=int, default=1, help="Batch size for validation")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--reward_data_path", type=str, required=True, help="Path to the reward data file")
    parser.add_argument("--template_path", type=str, required=True, help="Path to the Jinja template file")
    
    # Tokenizer max length and gradient accumulation
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum length for tokenized inputs")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="Number of gradient accumulation steps")
    parser.add_argument("--evaluation_steps", type=int, default=100, help="Number of steps between evaluations")

    # Learning rate scheduler arguments
    parser.add_argument("--warmup_epochs", type=float, default=0.1, help="Number of warmup epochs (as a fraction)")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate for cosine decay")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW optimizer")

    # LoRA-specific arguments
    parser.add_argument("--lora_r", type=int, default=8, help="Low-rank dimension for LoRA")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA scaling factor")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout rate")
    parser.add_argument("--target_modules", type=str, default="c_attn,q_proj,v_proj", help="Comma-separated list of target modules for LoRA")

    # Checkpoint arguments
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save the best LoRA checkpoint")
    parser.add_argument("--lora_checkpoint_path", type=str, default=None, help="Path to load LoRA checkpoint")

    # Wandb arguments
    parser.add_argument("--wandb_project", type=str, default="reward-model-training", help="Wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Wandb run name")

    args = parser.parse_args()
    trainer = RMTrainer(args)
    trainer.train()
