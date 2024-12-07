import argparse

from sotopia_rl import SotopiaPPOTrainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model with PPO using a reward model.")
    parser.add_argument("--model_name", type=str, default="/data/models/gemma-2-2b-it", help="Path to the model")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--ppo_data_path", type=str, required=True, help="Path to the reward data file")
    parser.add_argument("--template_path", type=str, required=True, help="Path to the Jinja template file")
    parser.add_argument("--ppo_epochs", type=int, default=4, help="Number of PPO epochs per update")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lam", type=float, default=0.95, help="GAE lambda for advantage estimation")
    parser.add_argument("--max_length", type=int, default=4096, help="Maximum length of input sequences")

    # LoRA-specific arguments
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank (dimension of the low-rank matrices)")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA scaling factor")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="Dropout rate for LoRA layers")
    parser.add_argument("--target_modules", type=str, default="c_attn,q_proj,v_proj", help="Comma-separated list of target modules for LoRA")

    args = parser.parse_args()
    trainer = SotopiaPPOTrainer(args)
    trainer.train()
