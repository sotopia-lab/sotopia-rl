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
from trl import get_kbit_device_map, PPOConfig, PPOTrainer
from accelerate import PartialState, Accelerator

import wandb
from sotopia_rl.data import PPODataset
os.environ['NCCL_P2P_DISABLE'] = '1' 
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

class SotopiaPPOTrainer:
    def __init__(self, args, accelerator: Accelerator):
        self.accelerator = accelerator
        self.args = args

        self._init_wandb()
        self._setup_tokenizer()
        self._setup_dataset()
        self._create_quantization_config()

        self._setup_generation_models()
        self._setup_classification_models()

        self.policy, self.ref_policy, self.reward_model, self.value_model = self.accelerator.prepare(
            self.policy, self.ref_policy, self.reward_model, self.value_model
        )
        self.policy = self.accelerator.unwrap_model(self.policy)
        self.ref_policy = self.accelerator.unwrap_model(self.ref_policy)
        self.reward_model = self.accelerator.unwrap_model(self.reward_model)
        self.value_model = self.accelerator.unwrap_model(self.value_model)

        self._setup_ppo_trainer()

        def save_model(self, output_dir: str, _internal_call: bool = False):
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
        wandb.init(
            project=self.args.wandb_project,
            name=self.args.wandb_run_name,
            config={k: v for k, v in vars(self.args).items() if isinstance(v, (int, float, str))}
        )

    def _setup_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name, padding_side="left")
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def _setup_dataset(self):
        with PartialState().local_main_process_first():
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
        self.quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    
    def _setup_generation_models(self):
        base_gen_ref = AutoModelForCausalLM.from_pretrained(
            self.args.model_name,
            torch_dtype='auto',
            quantization_config=self.quant_config,
            device_map=get_kbit_device_map(),
        )
        self.ref_policy = PeftModelForCausalLM.from_pretrained(
            base_gen_ref,
            self.args.ref_adapter_path,
            is_trainable=False,
            adapter_name="ref_adapter"
        )
        self.ref_policy.active_adapter = "ref_adapter"

        base_gen_policy = AutoModelForCausalLM.from_pretrained(
            self.args.model_name,
            torch_dtype='auto',
            quantization_config=self.quant_config,
            device_map=get_kbit_device_map(),
        )
        if self.args.use_lora_train_ppo:
            self.policy = PeftModelForCausalLM.from_pretrained(
                base_gen_policy,
                self.args.policy_adapter_path,
                is_trainable=True,
                adapter_name="policy_adapter"
            )
            self.policy.active_adapter = "policy_adapter"
        else:
            self.policy = base_gen_policy

        requires_grad_num = 0
        for name, param in self.policy.named_parameters():
            if param.requires_grad:
                requires_grad_num += 1
                print(name)
        print(f"Number of trainable parameters in policy: {requires_grad_num}")

        requires_grad_num = 0
        for name, param in self.ref_policy.named_parameters():
            if param.requires_grad:
                requires_grad_num += 1
        print(f"Number of trainable parameters in ref policy: {requires_grad_num}")

    def _setup_classification_models(self):
        base_reward_model = AutoModelForSequenceClassification.from_pretrained(
            self.args.model_name,
            torch_dtype='auto',
            num_labels=1,
            quantization_config=self.quant_config,
            device_map=get_kbit_device_map(),
        )
        self.reward_model = PeftModelForSequenceClassification.from_pretrained(
            base_reward_model,
            self.args.reward_adapter_path,
            is_trainable=False,
            adapter_name="reward_adapter"
        )
        self.reward_model.active_adapter = "reward_adapter"
        for name, param in self.reward_model.named_parameters():
            if self.reward_model.active_adapter in name:
                param.requires_grad = False

        base_value_model = AutoModelForSequenceClassification.from_pretrained(
            self.args.model_name,
            torch_dtype='auto',
            num_labels=1,
            quantization_config=self.quant_config,
            device_map=get_kbit_device_map(),
        )
        # VERY VERY IMPORTANT
        # specifically designed for PPO training, 
        # based on the get_reward function
        # it fill the input_ids paddings with 0s
        base_reward_model.config.pad_token_id = 0
        base_value_model.config.pad_token_id = 0

        if self.args.use_lora_train_ppo:
            self.value_model = PeftModelForSequenceClassification.from_pretrained(
                base_value_model,
                self.args.value_adapter_path,
                is_trainable=True,
                adapter_name="value_adapter"
            )
            self.value_model.active_adapter = "value_adapter"
        else:
            self.value_model = base_value_model
        
        requires_grad_num = 0
        for name, param in self.value_model.named_parameters():
            if param.requires_grad:
                requires_grad_num += 1
        print(f"Number of trainable parameters in value model: {requires_grad_num}")

        requires_grad_num = 0
        for name, param in self.reward_model.named_parameters():
            if param.requires_grad:
                requires_grad_num += 1
        print(f"Number of trainable parameters in reward model: {requires_grad_num}")

    def _setup_ppo_trainer(self):
        training_args = PPOConfig(
            per_device_train_batch_size=self.args.per_device_train_batch_size,
            per_device_eval_batch_size=self.args.per_device_eval_batch_size,
            num_mini_batches=self.args.num_mini_batches,
            local_rollout_forward_batch_size=self.args.local_rollout_forward_batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            num_train_epochs=self.args.num_train_epochs,
            num_ppo_epochs=self.args.num_ppo_epochs,
            learning_rate=self.args.learning_rate,
            output_dir=self.args.output_dir,
            gamma=self.args.gamma,
            lam=self.args.lam,
            save_steps=self.args.save_steps,
            missing_eos_penalty=self.args.missing_eos_penalty,
            ddp_find_unused_parameters=True,
            response_length=self.args.response_length,
            stop_token='eos',
        )

        self.ppo_trainer = PPOTrainer(
            args=training_args,
            model=self.policy,
            processing_class=self.tokenizer,
            ref_model=self.ref_policy,
            reward_model=self.reward_model,
            value_model=self.value_model,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
        )
        print("PPOtrainer setup complete")

    def train(self):
        try:
            print("Starting PPO training...")
            train_stats = self.ppo_trainer.train()
            return train_stats
        except Exception as e:
            print(f"Training error: {str(e)}")
            raise