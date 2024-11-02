import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, random_split
from torch.nn import MSELoss
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from jinja2 import Environment, FileSystemLoader
from data import RewardDataset
from tqdm import tqdm
import os
import wandb
import argparse

class RMTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_val_loss = float('inf')  # Initialize best validation loss

        # Initialize wandb
        wandb.init(
            project=args.wandb_project, 
            name=args.wandb_run_name,
            config={k: v for k, v in vars(args).items() if isinstance(v, (int, float, str))}
        )

        # Load the tokenizer and base model
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        self.tokenizer.model_max_length = args.max_length

        # Initialize model with LoRA
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules.split(",")
        )
        reward_model = AutoModelForCausalLMWithValueHead.from_pretrained(args.model_name)
        self.model = get_peft_model(reward_model, peft_config).to(self.device)

        # Load LoRA checkpoint if specified
        if args.lora_checkpoint_path:
            self.load_lora_checkpoint(args.lora_checkpoint_path)

        # Set up dataset and data loaders
        self.setup_dataloaders()

        # Initialize optimizer and learning rate scheduler
        self.optimizer = AdamW(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        self.lr_scheduler = self.create_lr_scheduler()

        # Initialize loss function
        self.loss_fn = MSELoss()



    def setup_dataloaders(self):
        # Load dataset and create train/val split
        env = Environment(loader=FileSystemLoader("/".join(self.args.template_path.split("/")[:-1])))
        template = env.get_template(self.args.template_path.split("/")[-1])
        dataset = RewardDataset(self.args.reward_data_path, self.tokenizer, template)
        
        train_size = int(0.95 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        self.train_loader = DataLoader(train_dataset, batch_size=self.args.train_batch_size, shuffle=True, collate_fn=dataset.collate_fn)
        self.val_loader = DataLoader(val_dataset, batch_size=self.args.val_batch_size, shuffle=False, collate_fn=dataset.collate_fn)
        self.steps_per_epoch = len(self.train_loader)

    def create_lr_scheduler(self):
        warmup_scheduler = LinearLR(
            self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=int(self.steps_per_epoch * self.args.warmup_epochs)
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer, T_max=self.args.num_epochs * self.steps_per_epoch, eta_min=self.args.min_lr
        )
        return SequentialLR(self.optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[int(self.steps_per_epoch * self.args.warmup_epochs)])

    def compute_loss(self, outputs, attention_masks, true_rewards):
        last_indices = (attention_masks.sum(dim=1) - 1).long()
        eos_values = outputs[torch.arange(outputs.size(0)), last_indices]
        return self.loss_fn(eos_values, true_rewards)

    def evaluate(self):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            val_progress = tqdm(enumerate(self.val_loader), total=len(self.val_loader), desc="Validating")
            for step, (input_ids, attention_masks, true_rewards) in val_progress:
                input_ids, attention_masks, true_rewards = (
                    input_ids.to(self.device),
                    attention_masks.to(self.device),
                    true_rewards.to(self.device),
                )
                _, _, outputs = self.model(input_ids, attention_mask=attention_masks, return_dict=True)
                loss = self.compute_loss(outputs, attention_masks, true_rewards)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(self.val_loader)
        wandb.log({"val_loss": avg_val_loss})

        # Save checkpoint if validation loss improves
        if avg_val_loss < self.best_val_loss:
            self.best_val_loss = avg_val_loss
            self.save_lora_checkpoint()

        return avg_val_loss

    def save_lora_checkpoint(self):
        os.makedirs(self.args.checkpoint_dir, exist_ok=True)
        
        # Create checkpoint directory
        checkpoint_path = os.path.join(self.args.checkpoint_dir, "best_lora_checkpoint")
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Save value head state dict
        torch.save(
            self.model.v_head.state_dict(), 
            os.path.join(checkpoint_path, "value_head_state_dict.pt")
        )
        
        # Save PEFT parameters
        peft_params = {
            name: param.data 
            for name, param in self.model.named_parameters() 
            if 'v_head' not in name and 'lora' in name
        }
        
        torch.save(
            peft_params, 
            os.path.join(checkpoint_path, "lora_parameters.pt")
        )
        
        print(f"LoRA and value head checkpoint saved at {checkpoint_path}")


    def load_lora_checkpoint(self, checkpoint_path):
        # Load value head state dict
        value_head_path = os.path.join(checkpoint_path, "value_head_state_dict.pt")
        if os.path.exists(value_head_path):
            self.model.v_head.load_state_dict(
                torch.load(value_head_path)
            )
        
        # Load LoRA parameters
        lora_params_path = os.path.join(checkpoint_path, "lora_parameters.pt")
        if os.path.exists(lora_params_path):
            lora_params = torch.load(lora_params_path)
            
            # Load LoRA parameters into the model
            model_state_dict = self.model.state_dict()
            for name, param in lora_params.items():
                if name in model_state_dict:
                    model_state_dict[name].copy_(param)
            
        print(f"LoRA and value head checkpoint loaded from {checkpoint_path}")

    def train(self):
        for epoch in range(self.args.num_epochs):
            self.model.train()
            total_loss = 0
            self.optimizer.zero_grad()

            train_progress = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f"Epoch {epoch + 1}/{self.args.num_epochs} - Training")
            for step, (input_ids, attention_masks, true_rewards) in train_progress:
                input_ids, attention_masks, true_rewards = (
                    input_ids.to(self.device),
                    attention_masks.to(self.device),
                    true_rewards.to(self.device),
                )
                _, _, outputs = self.model(input_ids, attention_mask=attention_masks)
                loss = self.compute_loss(outputs, attention_masks, true_rewards)
                loss = loss / self.args.accumulation_steps
                loss.backward()
                total_loss += loss.item()

                if (step + 1) % self.args.accumulation_steps == 0:
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                wandb.log({"train_loss": loss.item(), "learning_rate": self.lr_scheduler.get_last_lr()[0]})

                if (step + 1) % self.args.evaluation_steps == 0:
                    avg_val_loss = self.evaluate()
                    print(f"Step {step + 1} - Validation Loss: {avg_val_loss:.4f}")

            print(f"Epoch {epoch + 1}/{self.args.num_epochs} - Avg Train Loss: {total_loss / len(self.train_loader):.4f}")
        wandb.finish()

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
