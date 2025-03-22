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
from trl import PPOv2Config, PPOv2Trainer
from accelerate import Accelerator

import wandb
from sotopia_rl.data import PPODataset
os.environ['NCCL_P2P_DISABLE'] = '1' 


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
            if hasattr(self.model, "module"):
                model_to_save = self.model.module
            else:
                model_to_save = self.model
            if not hasattr(model_to_save, "policy"):
                model_to_save.policy = model_to_save
            model_to_save.save_pretrained(output_dir)
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
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name, padding_side="left", pad_token="<pad>")

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
        base_gen_model = AutoModelForCausalLM.from_pretrained(
            self.args.model_name,
            torch_dtype=torch.float32, # very important, otherwise NaN for RM
            quantization_config=self.quant_config,
            return_dict=True,
            device_map=None
        )

        if base_gen_model.config.pad_token_id is None:
            base_gen_model.config.pad_token_id = self.tokenizer.pad_token_id

        # Create generation config
        generation_config = GenerationConfig(
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.eos_token_id,
            max_length=self.args.max_length,
            do_sample=getattr(self.args, 'do_sample', True),
            temperature=getattr(self.args, 'temperature', 0.7),
            top_p=getattr(self.args, 'top_p', 0.9),
            repetition_penalty=getattr(self.args, 'repetition_penalty', 1.0),
            no_repeat_ngram_size=getattr(self.args, 'no_repeat_ngram_size', 0)
        )

        self.policy = PeftModelForCausalLM.from_pretrained(
            base_gen_model,
            self.args.policy_adapter_path,
            generation_config=generation_config
        )
        print("Policy model loaded/created")

        self.ref_policy = PeftModelForCausalLM.from_pretrained(
            base_gen_model,
            self.args.ref_adapter_path,
            generation_config=generation_config
        )
        self.ref_policy.eval()
        print("Reference policy model loaded/created")


    def _setup_classification_models(self):
        base_cls_model = AutoModelForSequenceClassification.from_pretrained(
            self.args.model_name,
            torch_dtype=torch.float32, # very important, otherwise NaN
            quantization_config=self.quant_config,
            num_labels=1,
            return_dict=True,
            device_map=None
        )

        if base_cls_model.config.pad_token_id is None:
            base_cls_model.config.pad_token_id = self.tokenizer.pad_token_id

        self.reward_model = PeftModelForSequenceClassification.from_pretrained(
            base_cls_model,
            self.args.reward_adapter_path,
            num_labels=1
        )
        self.reward_model.eval()
        print("Reward model loaded/created")

        self.value_model = PeftModelForSequenceClassification.from_pretrained(
            base_cls_model,
            self.args.value_adapter_path,
            num_labels=1
        )
        print("Value model loaded/created")

    def _setup_ppo_trainer(self):
        """Configure the PPO trainer"""
        # Get data collator if available


        # Configure PPO settings
        ppo_config = PPOv2Config(
            per_device_train_batch_size=self.args.per_device_train_batch_size,
            per_device_eval_batch_size=self.args.per_device_eval_batch_size,
            num_train_epochs=self.args.num_epochs,
            num_ppo_epochs=self.args.ppo_epochs,
            learning_rate=self.args.learning_rate,
            output_dir=self.args.checkpoint_dir,
            gamma=self.args.gamma,
            lam=self.args.lam,
            mini_batch_size=self.args.mini_batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            seed=self.args.seed,
            temperature=self.args.temperature,
            save_steps=self.args.save_steps,
            response_length=self.args.response_length, #important
            stop_token_id=self.tokenizer.eos_token_id, #important
            stop_token='eos', #important, just fill with pad after eos
            missing_eos_penalty=1.0,
            local_rollout_forward_batch_size=self.args.local_rollout_forward_batch_size,
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

        print("PPO trainer setup complete")

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
