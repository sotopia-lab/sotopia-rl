import os
import torch
from jinja2 import Environment, FileSystemLoader
from peft import PeftModelForCausalLM, PeftModelForSequenceClassification, get_peft_model, LoraConfig
from torch.utils.data import random_split
from transformers import (
    AutoModelForCausalLM, 
    AutoModelForSequenceClassification,
    AutoTokenizer, 
    BitsAndBytesConfig,
    GenerationConfig
)
from trl import PPOv2Config, PPOv2Trainer

import wandb
from sotopia_rl.data import PPODataset

class SotopiaPPOTrainer:
    def __init__(self, args):
        self.args = args
        self.device = f"cuda:0" if torch.cuda.is_available() else "cpu"
        
        # Initialize the training environment
        self._init_wandb()
        self._setup_tokenizer()
        self._setup_dataset()
        
        # Initialize quantization config for all models
        self.quant_config = self._create_quantization_config()
        
        # Load models - organized by type
        self._setup_generation_models()    # Policy and reference policy
        self._setup_classification_models() # Reward and value models
        
        # Setup the PPO trainer
        self._setup_ppo_trainer()
    
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
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
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
        
        val_ratio = getattr(self.args, 'val_ratio', 0.05)
        train_size = min(int(len(dataset) * (1 - val_ratio)), len(dataset) - 2)
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])
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
            torch_dtype=torch.float16,
            device_map=self.device,
            quantization_config=self.quant_config,
            return_dict=True,
        )
        base_gen_model.config.pad_token_id = self.tokenizer.eos_token_id
        
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
        
        # Default LoRA config for generation models
        gen_lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM"
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
            torch_dtype=torch.float16,
            device_map=self.device,
            quantization_config=self.quant_config,
            num_labels=1,
            return_dict=True
        )
        base_cls_model.config.pad_token_id = self.tokenizer.eos_token_id
        
        # Default LoRA config for classification models
        cls_lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type="SEQ_CLS"
        )
        
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
            per_device_train_batch_size=self.args.batch_size,
            per_device_eval_batch_size=self.args.batch_size,
            num_train_epochs=self.args.num_epochs,
            num_ppo_epochs=self.args.ppo_epochs,
            learning_rate=getattr(self.args, 'learning_rate', 1e-5),
            output_dir=self.args.checkpoint_dir,
            gamma=self.args.gamma,
            lam=self.args.lam,
            mini_batch_size=getattr(self.args, 'mini_batch_size', 1),
            gradient_accumulation_steps=getattr(self.args, 'gradient_accumulation_steps', 1),
            seed=getattr(self.args, 'seed', 42),
            temperature=getattr(self.args, 'temperature', 0.7),
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
