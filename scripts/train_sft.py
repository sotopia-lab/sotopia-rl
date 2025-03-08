import argparse
from sotopia_rl import SotopiaSFTTrainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a language model using SFT with LoRA.")
    
    # Basic training parameters
    parser.add_argument("--local-rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Model name or path")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate for scheduler")
    parser.add_argument("--train_batch_size", type=int, default=2, help="Training batch size")
    parser.add_argument("--val_batch_size", type=int, default=2, help="Validation batch size")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--sft_data_path", type=str, required=True, help="Path to SFT dataset")
    parser.add_argument("--template_path", type=str, required=True, help="Path to the Jinja template file")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--evaluation_steps", type=int, default=100, help="Evaluation interval in steps")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    
    # LoRA-specific arguments
    parser.add_argument("--warmup_epochs", type=float, default=0.1, help="Number of warmup epochs (as a fraction)")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA for fine-tuning")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--target_modules", type=str, default="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj", 
                        help="Target modules for LoRA (comma-separated)")
    
    # Checkpoint and Wandb arguments
    parser.add_argument("--checkpoint_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--lora_checkpoint_path", type=str, default=None, help="Path to load LoRA checkpoint")
    parser.add_argument("--wandb_project", type=str, default="sft-project", help="Wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default="sft-run", help="Wandb run name")
    
    # QLoRA and optimization arguments
    parser.add_argument("--use_qlora", action="store_true", help="Use QLoRA (4-bit) for model loading")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Use gradient checkpointing to save memory")
    parser.add_argument("--flash_attention", action="store_true", help="Use flash attention for faster training")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 precision")
    
    # Additional parameter additions
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging interval in steps")
    parser.add_argument("--save_total_limit", type=int, default=3, help="Maximum number of checkpoints to keep")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm for clipping")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", 
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
                        help="Type of learning rate scheduler")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize and train
    trainer = SotopiaSFTTrainer(args)
    trainer.train()