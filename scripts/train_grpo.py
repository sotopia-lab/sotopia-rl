import argparse
import os
os.environ["TRANSFORMERS_NO_COMPILE"] = "1"
import argparse
from sotopia_rl import SotopiaGRPOTrainer
from accelerate import Accelerator

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model with GRPO using a reward model.")
    
    parser.add_argument("--model_name", type=str, default="/data/models/gemma-2-2b-it",
                        help="Path to the model")

    parser.add_argument("--per_device_train_batch_size", type=int, default=1,
                        help="Batch size per device for training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1,
                        help="Batch size per device for evaluation")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--num_grpo_epochs", type=int, default=4,
                        help="Number of GRPO epochs per update")
    parser.add_argument("--learning_rate", type=float, default=5e-6,
                        help="Learning rate for optimizer")
    parser.add_argument("--max_length", type=int, default=4096,
                        help="Maximum length of input sequences")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of steps to accumulate gradients before performing an update")
    parser.add_argument("--val_ratio", type=float, default=0.05,
                        help="Ratio of validation data")
    parser.add_argument("--response_length", type=int, default=128,
                        help="Maximum length of generated responses")

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
    parser.add_argument("--grpo_data_path", type=str, required=True,
                        help="Path to the reward data file")
    parser.add_argument("--template_path", type=str, required=True,
                        help="Path to the Jinja template file")
    parser.add_argument("--output_dir", type=str, default="checkpoints",
                        help="Directory to save the best LoRA checkpoint")
    parser.add_argument("--save_steps", type=int, default=5,
                        help="Number of steps between saving checkpoints")

    # Logging parameters
    parser.add_argument("--wandb_project", type=str, default="grpo-model-training",
                        help="Wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Wandb run name")
 
    args = parser.parse_args()
    accelerator = Accelerator()
    trainer = SotopiaGRPOTrainer(args, accelerator)
    trainer.train()
