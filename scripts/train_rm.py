import argparse

from sotopia_rl import SotopiaRMTrainer

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
    parser.add_argument("--max_length", type=int, default=4096, help="Maximum length for tokenized inputs")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="Number of gradient accumulation steps")
    parser.add_argument("--evaluation_steps", type=int, default=100, help="Number of steps between evaluations")

    # Learning rate scheduler arguments
    parser.add_argument("--warmup_epochs", type=float, default=0.1, help="Number of warmup epochs (as a fraction)")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate for cosine decay")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW optimizer")

    # LoRA-specific arguments
    parser.add_argument("--lora_r", type=int, default=16, help="Low-rank dimension for LoRA")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA scaling factor")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout rate")
    parser.add_argument("--target_modules", type=str, default="q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj",
                        help="Comma-separated list of target modules for LoRA")

    # Checkpoint arguments
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory to save the best LoRA checkpoint")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to load LoRA checkpoint")

    # Wandb arguments
    parser.add_argument("--wandb_project", type=str, default="reward-model-training", help="Wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Wandb run name")

    args = parser.parse_args()
    trainer = SotopiaRMTrainer(args)
    trainer.train()
