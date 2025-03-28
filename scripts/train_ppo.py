import argparse
import os
import argparse
from accelerate import Accelerator
from sotopia_rl import SotopiaPPOTrainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model with PPO using a reward model.")
    

    # Base model arguments
    parser.add_argument("--model_name", type=str, default="/data/models/gemma-2-2b-it",
                        help="Path to the model")
    parser.add_argument("--reward_model_name", type=str, default="/data/models/gemma-2-2b-it",
                        help="Path to the reward model")
    parser.add_argument("--value_model_name", type=str, default=None,
                        help="Path to the value model (defaults to model_name if not specified)")

    # Training parameters
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,
                        help="Batch size per device for training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1,
                        help="Batch size per device for evaluation")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--ppo_epochs", type=int, default=4,
                        help="Number of PPO epochs per update")
    parser.add_argument("--learning_rate", type=float, default=5e-6,
                        help="Learning rate for optimizer")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--lam", type=float, default=0.95,
                        help="GAE lambda for advantage estimation")
    parser.add_argument("--max_length", type=int, default=4096,
                        help="Maximum length of input sequences")
    parser.add_argument("--num_mini_batches", type=int, default=1,
                        help="Mini batch size for PPO updates")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of steps to accumulate gradients before performing an update")
    parser.add_argument("--val_ratio", type=float, default=0.05,
                        help="Ratio of validation data")
    parser.add_argument("--response_length", type=int, default=128,
                        help="Maximum length of generated responses")
    parser.add_argument("--local_rollout_forward_batch_size", type=int, default=16,
                        help="Batch size for local rollout forward pass")

    # LoRA parameters
    parser.add_argument("--use_lora", action="store_true",
                        help="Use LoRA for model loading")
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA rank (dimension of the low-rank matrices)")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA scaling factor")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="Dropout rate for LoRA layers")
    parser.add_argument("--target_modules", type=str, default="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj",
                        help="Comma-separated list of target modules for LoRA on causal models")
    parser.add_argument("--sequence_target_modules", type=str, default=None,
                        help="Comma-separated list of target modules for LoRA on sequence classification models")

    # Quantization parameters
    parser.add_argument("--policy_use_qlora", action="store_true",
                        help="Use QLoRA (4-bit) for policy model loading")
    parser.add_argument("--reward_use_qlora", action="store_true",
                        help="Use QLoRA (4-bit) for reward model loading")
    parser.add_argument("--value_use_qlora", action="store_true",
                        help="Use QLoRA (4-bit) for value model loading")
    parser.add_argument("--ref_use_qlora", action="store_true",
                        help="Use QLoRA (4-bit) for reference model loading")

    # Adapter parameters
    parser.add_argument("--policy_adapter_path", type=str, default=None,
                        help="Path to policy model adapter")
    parser.add_argument("--reward_adapter_path", type=str, default=None,
                        help="Path to reward model adapter")
    parser.add_argument("--value_adapter_path", type=str, default=None,
                        help="Path to value model adapter")
    parser.add_argument("--ref_adapter_path", type=str, default=None,
                        help="Path to reference model adapter")

    # Device assignment
    parser.add_argument("--policy_gpu", type=int, default=0,
                        help="GPU ID for policy model")
    parser.add_argument("--reward_gpu", type=int, default=None,
                        help="GPU ID for reward model")
    parser.add_argument("--value_gpu", type=int, default=None,
                        help="GPU ID for value model")
    parser.add_argument("--ref_gpu", type=int, default=None,
                        help="GPU ID for reference model")

    # Generation parameters
    parser.add_argument("--do_sample", action="store_true",
                        help="Whether to use sampling for generation")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p (nucleus) sampling parameter")
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                        help="Repetition penalty for generation")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0,
                        help="Size of n-grams to avoid repeating")

    # Model behavior parameters
    parser.add_argument("--freeze_value_model", action="store_true",
                        help="Whether to freeze the value model during training")

    # Data and checkpoint paths
    parser.add_argument("--ppo_data_path", type=str, required=True,
                        help="Path to the reward data file")
    parser.add_argument("--template_path", type=str, required=True,
                        help="Path to the Jinja template file")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory to save the best LoRA checkpoint")
    parser.add_argument("--save_steps", type=int, default=50,
                        help="Number of steps between saving checkpoints")

    # Logging parameters
    parser.add_argument("--wandb_project", type=str, default="ppo-model-training",
                        help="Wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Wandb run name")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
 

    args = parser.parse_args()
    accelerator = Accelerator()

    # Initialize trainer and start training
    trainer = SotopiaPPOTrainer(args, accelerator)
    trainer.train()
