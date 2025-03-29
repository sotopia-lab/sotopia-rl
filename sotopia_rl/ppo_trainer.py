import os

import torch
from jinja2 import Environment, FileSystemLoader
from peft import PeftModelForCausalLM, PeftModelForSequenceClassification
from torch.utils.data import random_split
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)
from trl import PPOv2Config, get_kbit_device_map
from .ppo_trainer_src import PPOv2Trainer
from accelerate import Accelerator

import wandb
from sotopia_rl.data import PPODataset
os.environ['NCCL_P2P_DISABLE'] = '1' 
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

class SotopiaPPOTrainer:
    def __init__(self, args, accelerator):
        self.args = args
        self.accelerator = accelerator
        self.device = accelerator.device

        # Initialize the training environment
        self._init_wandb()
        self._setup_tokenizer()
        self._setup_dataset()

        # Initialize quantization config for all models
        self.quant_config = self._create_quantization_config()

        # Load models - organized by type
        self._setup_generation_models()    # Policy and reference policy
        self._setup_classification_models() # Reward and value models

        self.policy, self.ref_policy, self.reward_model, self.value_model = self.accelerator.prepare(
            self.policy, self.ref_policy, self.reward_model, self.value_model
        )
        self.policy = self.accelerator.unwrap_model(self.policy)
        self.ref_policy = self.accelerator.unwrap_model(self.ref_policy)
        self.reward_model = self.accelerator.unwrap_model(self.reward_model)
        self.value_model = self.accelerator.unwrap_model(self.value_model)

        self.policy.to(self.device)
        self.ref_policy.to(self.device)
        self.reward_model.to(self.device)
        self.value_model.to(self.device)

        # Setup the PPO trainer
        self._setup_ppo_trainer()

        def save_model(self, output_dir: str, _internal_call: bool = False):
            print(self.model)
            print(self.model.policy)
            if hasattr(self.model, "policy"):
                model_to_save = self.model.policy
            elif hasattr(self.model, "module") and hasattr(self.model.module, "policy"):
                model_to_save = self.model.module.policy
            else:
                model_to_save = self.model
            model_to_save.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            print(f"Model saved to {output_dir}")

        self.ppo_trainer.save_model = save_model.__get__(self.ppo_trainer, type(self.ppo_trainer))

    def _init_wandb(self):
        """Initialize Weights & Biases logging"""
        wandb.init(
            project=self.args.wandb_project,
            name=self.args.wandb_run_name,
            config={k: v for k, v in vars(self.args).items() if isinstance(v, (int, float, str))}
        )

    def _setup_tokenizer(self):
        """Load and configure tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name, padding_side="left")

    def _setup_dataset(self):
        """Prepare training and validation datasets"""
        # Load template for formatting prompts
        template_dir = "/".join(self.args.template_path.split("/")[:-1])
        template_file = self.args.template_path.split("/")[-1]
        env = Environment(loader=FileSystemLoader(template_dir))
        template = env.get_template(template_file)

        # Create and split dataset
        dataset = PPODataset(
            self.args.ppo_data_path,
            self.tokenizer,
            template,
            max_length=self.args.max_length
        )
        print(f"dataset: {len(dataset)}")
        
        generator = torch.Generator().manual_seed(42)
        val_ratio = getattr(self.args, 'val_ratio', 0.05)
        train_size = min(int(len(dataset) * (1 - val_ratio)), len(dataset) - 2)
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
        print(f"Dataset split: {len(self.train_dataset)} train, {len(self.val_dataset)} validation")

    def _create_quantization_config(self):
        """Create 4-bit quantization config for memory efficiency"""
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    
    def _setup_generation_models(self):
        # Load a single base model
        base_gen_model = AutoModelForCausalLM.from_pretrained(
            self.args.model_name,
            torch_dtype='auto',
            quantization_config=self.quant_config,
            device_map=get_kbit_device_map(),
        )
        base_gen_model.config.pad_token_id = self.tokenizer.pad_token_id

        self.ref_policy = PeftModelForCausalLM.from_pretrained(
            base_gen_model,
            self.args.ref_adapter_path,
        )
        self.ref_policy.merge_and_unload()
        self.ref_policy.eval()
        
        self.policy = AutoModelForCausalLM.from_pretrained(
            self.args.model_name,
            torch_dtype='auto',
            quantization_config=self.quant_config,
            device_map=get_kbit_device_map(),
        )
        self.policy.config.pad_token_id = self.tokenizer.pad_token_id

        
        # Count trainable parameters
        requires_grad_num = 0
        for name, param in self.policy.named_parameters():
            if param.requires_grad:
                requires_grad_num += 1
        print(f"Number of trainable parameters in policy: {requires_grad_num}")

        requires_grad_num = 0
        for name, param in self.ref_policy.named_parameters():
            if param.requires_grad:
                requires_grad_num += 1
                print(name)
        print(f"Number of trainable parameters in ref policy: {requires_grad_num}")

    def _setup_classification_models(self):
        base_cls_model = AutoModelForSequenceClassification.from_pretrained(
            self.args.model_name,
            torch_dtype='auto',
            num_labels=1,
            quantization_config=self.quant_config,
            return_dict=True,
            device_map=get_kbit_device_map(),
        )
        base_cls_model.config.pad_token_id = 0
        self.reward_model = PeftModelForSequenceClassification.from_pretrained(
            base_cls_model,
            self.args.reward_adapter_path,
        )
        self.reward_model.merge_and_unload()
        for name, param in self.reward_model.named_parameters():
            if 'score' in name:
                param.requires_grad = False

        self.value_model = AutoModelForSequenceClassification.from_pretrained(
            self.args.model_name,
            torch_dtype='auto',
            num_labels=1,
            quantization_config=self.quant_config,
            return_dict=True,
            device_map=get_kbit_device_map(),
        )

        # Count trainable parameters
        requires_grad_num = 0
        for name, param in self.value_model.named_parameters():
            if param.requires_grad:
                requires_grad_num += 1
        print(f"Number of trainable parameters in value model: {requires_grad_num}")

        requires_grad_num = 0
        for name, param in self.reward_model.named_parameters():
            if param.requires_grad:
                requires_grad_num += 1
                print(name)
        print(f"Number of trainable parameters in reward model: {requires_grad_num}")

    def _setup_ppo_trainer(self):
        """Configure the PPO trainer"""
        # Configure PPO settings
        ppo_config = PPOv2Config(
            per_device_train_batch_size=self.args.per_device_train_batch_size,
            per_device_eval_batch_size=self.args.per_device_eval_batch_size,
            num_mini_batches=self.args.num_mini_batches,
            local_rollout_forward_batch_size=self.args.local_rollout_forward_batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            num_train_epochs=self.args.num_epochs,
            num_ppo_epochs=self.args.ppo_epochs,
            learning_rate=self.args.learning_rate,
            output_dir=self.args.checkpoint_dir,
            gamma=self.args.gamma,
            lam=self.args.lam,
            seed=self.args.seed,
            temperature=self.args.temperature,
            save_steps=5,
            response_length=self.args.response_length, #important
            stop_token_id=198, #very important, 198 is \n, we need to stop at EOS + \n because sequence classification jinja
            missing_eos_penalty=1.0,
        )

        # Create the TRL PPO trainer
        self.ppo_trainer = PPOv2Trainer(
            policy=self.policy,
            ref_policy=self.ref_policy,
            reward_model=self.reward_model,
            value_model=self.value_model,
            config=ppo_config,
            tokenizer=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
        )
        print("PPOtrainer setup complete")

    def train(self):
        """Run PPO training loop and save checkpoints"""
        try:
            print("Starting PPO training...")
            train_stats = self.ppo_trainer.train()

            # Save final checkpoint
            final_checkpoint_dir = os.path.join(self.args.checkpoint_dir, "final_checkpoint")
            self.save_checkpoint(final_checkpoint_dir)

            return train_stats
        except Exception as e:
            print(f"Training error: {str(e)}")
            raise

    def save_checkpoint(self, checkpoint_path):
        """Save all model adapters"""
        os.makedirs(checkpoint_path, exist_ok=True)

        # Save policy adapter
        policy_dir = os.path.join(checkpoint_path, "policy")
        os.makedirs(policy_dir, exist_ok=True)
        self.policy.save_pretrained(policy_dir)

        # Save value model adapter
        value_dir = os.path.join(checkpoint_path, "value")
        os.makedirs(value_dir, exist_ok=True)
        self.value_model.save_pretrained(value_dir)

        # Save reward model adapter if it was trained
        reward_dir = os.path.join(checkpoint_path, "reward")
        os.makedirs(reward_dir, exist_ok=True)
        self.reward_model.save_pretrained(reward_dir)

        # Save tokenizer
        self.tokenizer.save_pretrained(checkpoint_path)

        print(f"Saved checkpoint to {checkpoint_path}")
