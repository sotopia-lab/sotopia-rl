import argparse
import os
from sotopia_rl import SotopiaSFTTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a reward model using SFT with LoRA.")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Model name or path")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--train_batch_size", type=int, default=2, help="Training batch size")
    parser.add_argument("--val_batch_size", type=int, default=2, help="Validation batch size")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--sft_data_path", type=str, required=True, help="Path to SFT data")
    parser.add_argument("--template_path", type=str, required=True, help="Path to the Jinja template file")
    parser.add_argument("--max_length", type=int, default=4096, help="Max sequence length")
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
    parser.add_argument("--use_qlora", action="store_true", help="Use QLoRA (4-bit) for model loading.")
    
    # DeepSpeed and distributed training arguments
    parser.add_argument("--use_distributed", action="store_true", help="Enable distributed training")
    parser.add_argument("--deepspeed", action="store_true", help="Enable DeepSpeed")
    parser.add_argument("--deepspeed_config", type=str, default=None, 
                        help="Path to DeepSpeed config file")
    parser.add_argument("--zero_stage", type=int, default=2, 
                        help="ZeRO optimization stage (0, 1, 2, 3)")
    parser.add_argument("--offload_optimizer", action="store_true", 
                        help="Offload optimizer states to CPU")
    parser.add_argument("--offload_param", action="store_true", 
                        help="Offload parameters to CPU")
    parser.add_argument("--gradient_checkpointing", action="store_true", 
                        help="Enable gradient checkpointing")
    
    args = parser.parse_args()
    
    # Auto-setup distributed training if using DeepSpeed
    if args.deepspeed:
        args.use_distributed = True
        
        # Create DeepSpeed config if not provided
        if args.deepspeed_config is None:
            import json
            
            ds_config_path = os.path.join(args.checkpoint_dir, "ds_config.json")
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            
            # Create a dynamic DeepSpeed config based on args
            ds_config = {
                "train_batch_size": "auto",
                "train_micro_batch_size_per_gpu": args.train_batch_size,
                "gradient_accumulation_steps": args.accumulation_steps,
                
                "optimizer": {
                    "type": "AdamW",
                    "params": {
                        "lr": args.learning_rate,
                        "betas": [0.9, 0.999],
                        "eps": 1e-8,
                        "weight_decay": args.weight_decay
                    }
                },
                
                "scheduler": {
                    "type": "WarmupLR",
                    "params": {
                        "warmup_min_lr": "auto",
                        "warmup_max_lr": args.learning_rate,
                        "warmup_num_steps": "auto"
                    }
                },
                
                "fp16": {
                    "enabled": True,
                    "auto_cast": True,
                    "loss_scale": 0,
                    "initial_scale_power": 16,
                    "loss_scale_window": 1000,
                    "hysteresis": 2,
                    "min_loss_scale": 1
                },
                
                "zero_optimization": {
                    "stage": args.zero_stage,
                    "allgather_partitions": True,
                    "allgather_bucket_size": 5e8,
                    "overlap_comm": True,
                    "reduce_scatter": True,
                    "reduce_bucket_size": 5e8,
                    "contiguous_gradients": True
                },
                
                "gradient_clipping": 1.0,
                "steps_per_print": 100,
                "wall_clock_breakdown": False
            }
            
            # Add optimizer offloading if requested
            if args.offload_optimizer:
                ds_config["zero_optimization"]["offload_optimizer"] = {
                    "device": "cpu",
                    "pin_memory": True
                }
            
            # Add parameter offloading if requested
            if args.offload_param:
                ds_config["zero_optimization"]["offload_param"] = {
                    "device": "cpu",
                    "pin_memory": True
                }
            
            # Write the config to a file
            with open(ds_config_path, 'w') as f:
                json.dump(ds_config, f, indent=4)
            
            args.deepspeed_config = ds_config_path
            print(f"Created DeepSpeed config at {ds_config_path}")
    
    # Configure local_rank automatically if needed
    if args.use_distributed and args.local_rank == -1:
        if 'LOCAL_RANK' in os.environ:
            args.local_rank = int(os.environ['LOCAL_RANK'])
        else:
            print("Warning: --use_distributed is set but no local_rank detected. Setting to 0.")
            args.local_rank = 0
    
    trainer = SotopiaSFTTrainer(args)
    trainer.train()