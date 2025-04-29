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
from peft import prepare_model_for_kbit_training
from trl.trainer.utils import disable_dropout_in_model
from accelerate import PartialState, Accelerator
import copy

import wandb
from sotopia_rl.data import PPODataset
os.environ['NCCL_P2P_DISABLE'] = '1' 
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

class SotopiaPPOTrainer:
    def __init__(self, args):
        self.args = args

        self._init_wandb()
        self._setup_tokenizer()
        self._setup_dataset()
        self._create_quantization_config()

        self._setup_generation_models()
        self._setup_classification_models()
        self._setup_ppo_trainer()

        for m in [self.policy]:
            m.config.use_cache = False
        for m in [self.value_model, self.reward_model]:
            m.config.use_cache = False

        def save_model(self, output_dir: str, _internal_call: bool = False):
            if hasattr(self.model, "policy"):
                policy_to_save = self.model.policy
                value_to_save = self.model.value_model
            elif hasattr(self.model, "module"):
                policy_to_save = self.model.module.policy
                value_to_save = self.model.module.value_model
            else:
                raise ValueError("Model does not have 'policy' or 'module' attribute")
            policy_to_save.save_pretrained(output_dir)
            value_to_save.save_pretrained(output_dir)
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
                data_path=self.args.ppo_data_path,
                tokenizer=self.tokenizer,
                template=template,
                max_length=self.args.max_length
            )
            print(f"dataset: {len(dataset)}")
            
            generator = torch.Generator().manual_seed(42)
            val_ratio = getattr(self.args, 'val_ratio', 0.05)
            train_size = min(int(len(dataset) * (1 - val_ratio)), len(dataset) - 10)
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
        self.ref_policy = AutoModelForCausalLM.from_pretrained(
            self.args.model_name,
            torch_dtype='auto',
        )

        
        if self.args.use_lora_train_ppo:
            self.base_gen_policy = AutoModelForCausalLM.from_pretrained(
                self.args.model_name,
                torch_dtype='auto',
            )
            self.policy = PeftModelForCausalLM.from_pretrained(
                self.base_gen_policy,
                self.args.policy_adapter_path,
                is_trainable=True,
                adapter_name="policy_adapter"
            )
        else:
            self.policy = AutoModelForCausalLM.from_pretrained(
                self.args.model_name,
                torch_dtype='auto',
            )
        self.ref_policy.config.pad_token_id = self.tokenizer.pad_token_id
        self.policy.config.pad_token_id = self.tokenizer.pad_token_id

        requires_grad_num = 0
        for name, param in self.policy.named_parameters():
            print(name, param.requires_grad)
            if param.requires_grad:
                requires_grad_num += 1
        print(f"Number of trainable parameters in policy: {requires_grad_num}")


        #for name, param in self.policy.named_parameters():
        #    if self.policy.active_adapter in name:
        #        param.requires_grad = False
        #requires_grad_num = 0
        #for name, param in self.ref_policy.named_parameters():
        #    if param.requires_grad:
        #        requires_grad_num += 1
        #print(f"Number of trainable parameters in ref policy: {requires_grad_num}")

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

        for name, param in self.reward_model.named_parameters():
            if self.reward_model.active_adapter in name:
                param.requires_grad = False

        if self.args.use_lora_train_ppo:
            base_value_model = AutoModelForSequenceClassification.from_pretrained(
                self.args.model_name,
                torch_dtype='auto',
                num_labels=1,
            )
            self.value_model = PeftModelForSequenceClassification.from_pretrained(
                base_value_model,
                self.args.value_adapter_path,
                is_trainable=True,
                adapter_name="value_adapter"
            )
        else:
            self.value_model = AutoModelForSequenceClassification.from_pretrained(
                self.args.model_name,
                torch_dtype='auto',
                num_labels=1,
            )
        
        # need to set this with not None results
        self.value_model.config.pad_token_id = self.tokenizer.pad_token_id
        self.reward_model.config.pad_token_id = self.tokenizer.pad_token_id
        
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
            kl_estimator='k3',
            vf_coef=1e-3,
            kl_coef=0.05,
        )

        self.ppo_trainer = PPOTrainer(
            args=training_args,
            model=self.policy,
            ref_model=self.ref_policy,
            processing_class=self.tokenizer,
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