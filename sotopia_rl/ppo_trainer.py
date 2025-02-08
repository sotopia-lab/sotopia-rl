import torch
import wandb
from jinja2 import Environment, FileSystemLoader
from peft import LoraConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from sotopia_rl.data import PPODataset
import os


class SotopiaPPOTrainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={k: v for k, v in vars(args).items() if isinstance(v, (int, float, str))}
        )

        # Initialize tokenizer and reward model (reward model remains frozen)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        # self.reward_model = AutoModelForCausalLMWithValueHead.from_pretrained(args.model_name)
        self.reward_model = AutoModelForCausalLMWithValueHead.from_pretrained(args.reward_model_name)
        self.reward_model.eval()  # Freeze the reward model
        self.reward_model.to(self.device)

        # Configure and initialize the PPO policy model with LoRA
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules.split(",")  # Specify modules for LoRA
        )
        self.policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(args.model_name, peft_config=peft_config)
        self.policy_model.to(self.device)

        self.ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(args.model_name)
        self.ref_model.eval()
        self.ref_model.to(self.device)

        # Set up PPO configuration and trainer
        ppo_config = PPOConfig(
            batch_size=args.batch_size,
            ppo_epochs=args.ppo_epochs,
            gamma=args.gamma,
            lam=args.lam,
            mini_batch_size=1,
            gradient_accumulation_steps=1,
            seed=42  # Set the seed value here
        )

        # Load dataset and set up template
        env = Environment(loader=FileSystemLoader("/".join(args.template_path.split("/")[:-1])))
        self.template = env.get_template(args.template_path.split("/")[-1])
        self.dataset = PPODataset(args.ppo_data_path, self.tokenizer, self.template, max_length=args.max_length)

        # Initialize the PPOTrainer with model, ref_model (copy of policy_model), and config
        self.ppo_trainer = PPOTrainer(
            model=self.policy_model,
            ref_model=self.ref_model,  # Reference model to stabilize updates
            config=ppo_config,
            tokenizer=self.tokenizer,
            data_collator=self.dataset.collate_fn
        )

    def train(self):
        train_loader = DataLoader(self.dataset, batch_size=self.args.batch_size, shuffle=True,
                                  collate_fn=self.dataset.collate_fn)

        for epoch in range(self.args.num_epochs):
            epoch_loss = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.args.num_epochs}"):
                input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
                input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)

                # Generate actions (outputs) and compute rewards
                generated_output = self.policy_model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                                              max_length=4096)

                # Obtain rewards by evaluating generated outputs with the reward model
                batch_rewards = self.compute_rewards(generated_output, attention_mask)

                # Perform PPO update
                if isinstance(input_ids, torch.Tensor):
                    input_ids = list(input_ids.unbind(0))
                if isinstance(generated_output, torch.Tensor):
                    generated_output = list(generated_output.unbind(0))
                if isinstance(batch_rewards, torch.Tensor):
                    if batch_rewards.dim() == 0:
                        batch_rewards = [batch_rewards]
                    else:
                        batch_rewards = list(batch_rewards.unbind(0))

                loss = self.ppo_trainer.step(input_ids, generated_output, batch_rewards)
                epoch_loss += loss["ppo/loss/total"]
            print(f"Epoch {epoch + 1} - PPO Loss: {epoch_loss / len(train_loader):.4f}")

            checkpoint_dir = os.path.join(self.args.checkpoint_dir, f"checkpoint-epoch-{epoch + 1}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            self.policy_model.save_pretrained(checkpoint_dir)

        final_checkpoint_dir = os.path.join(self.args.checkpoint_dir, "final_checkpoint")
        os.makedirs(final_checkpoint_dir, exist_ok=True)
        self.policy_model.save_pretrained(final_checkpoint_dir)
        print(f"Final checkpoint saved at {final_checkpoint_dir}")

    def compute_rewards(self, generated_output, attention_mask):
        """Compute rewards using the frozen reward model."""
        with torch.no_grad():
            _, _, output_values = self.reward_model(generated_output, attention_mask=attention_mask)
        rewards = output_values.mean(dim=-1)
        if rewards.dim() > 1:
            rewards = rewards.mean(dim=-1)
        if rewards.dim() == 0:
            rewards = rewards.unsqueeze(0)
        return rewards

    def save_checkpoint(self, checkpoint_path):
        os.makedirs(checkpoint_path, exist_ok=True)
        self.policy_model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
