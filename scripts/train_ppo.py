import argparse
import os
import argparse
from sotopia_rl import SotopiaPPOTrainer
from accelerate import Accelerator

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model with PPO using a reward model.")
    
    parser.add_argument("--model_name", type=str, default="/data/models/gemma-2-2b-it",
                        help="Path to the model")

    parser.add_argument("--per_device_train_batch_size", type=int, default=1,
                        help="Batch size per device for training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1,
                        help="Batch size per device for evaluation")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--num_ppo_epochs", type=int, default=4,
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

    # Adapter parameters
    parser.add_argument("--policy_adapter_path", type=str, default=None,
                        help="Path to policy model adapter")
    parser.add_argument("--reward_adapter_path", type=str, default=None,
                        help="Path to reward model adapter")
    parser.add_argument("--value_adapter_path", type=str, default=None,
                        help="Path to value model adapter")
    parser.add_argument("--ref_adapter_path", type=str, default=None,
                        help="Path to reference model adapter")

    # Data and checkpoint paths
    parser.add_argument("--ppo_data_path", type=str, required=True,
                        help="Path to the reward data file")
    parser.add_argument("--template_path", type=str, required=True,
                        help="Path to the Jinja template file")
    parser.add_argument("--output_dir", type=str, default="checkpoints",
                        help="Directory to save the best LoRA checkpoint")
    parser.add_argument("--save_steps", type=int, default=5,
                        help="Number of steps between saving checkpoints")
    parser.add_argument("--missing_eos_penalty", type=float, default=1.0,
                        help="Penalty for missing EOS token in generated")

    # Logging parameters
    parser.add_argument("--wandb_project", type=str, default="ppo-model-training",
                        help="Wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Wandb run name")
    
    parser.add_argument("--use_lora_train_ppo", action="store_true",
                        help="Use LoRA for training PPO")
 
    args = parser.parse_args()
    accelerator = Accelerator()
    trainer = SotopiaPPOTrainer(args, accelerator)
    trainer.train()
