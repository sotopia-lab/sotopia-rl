import os
import json
import torch
import wandb
from torch.utils.data import random_split, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModelForCausalLM, PeftModelForSequenceClassification
from jinja2 import Environment, FileSystemLoader
from trl import get_kbit_device_map, GRPOConfig, GRPOTrainer, get_peft_config
from accelerate import PartialState, Accelerator
from sotopia_rl.data import GRPODataset
from peft import LoraConfig, TaskType, get_peft_model

os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

class SotopiaGRPOTrainer:
    def __init__(self, args, accelerator: Accelerator):
        self.args = args
        self.accelerator = accelerator

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
        print("\n=== Trainable parameters in policy ===")
        for name, param in self.policy.named_parameters():
            if param.requires_grad:
                print(f"{name}: {param.shape}")
        
        print("\n=== Trainable parameters in reward model ===")
        for name, param in self.reward_model.named_parameters():
            if param.requires_grad:
                print(f"{name}: {param.shape}")

        print("\n=== Trainable parameters in value model ===")
        for name, param in self.value_model.named_parameters():
            if param.requires_grad:
                print(f"{name}: {param.shape}")

        num_params = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
        print(f"Total trainable parameters in policy: {num_params}")

        self._setup_grpo_trainer()

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

        self.grpo_trainer.save_model = save_model.__get__(self.grpo_trainer, type(self.grpo_trainer))
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
        env = Environment(loader=FileSystemLoader("/".join(self.args.template_path.split("/")[:-1])))
        template = env.get_template(self.args.template_path.split("/")[-1])
        dataset = GRPODataset(
            data_path=self.args.grpo_data_path,
            tokenizer=self.tokenizer,
            template=template,
            max_length=self.args.max_length
        )
        
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
        base_gen_policy = AutoModelForCausalLM.from_pretrained(
            self.args.model_name,
            torch_dtype='auto',
            quantization_config=self.quant_config,
            device_map=get_kbit_device_map(),
        )

        base_gen_ref = AutoModelForCausalLM.from_pretrained(
            self.args.model_name,
            torch_dtype='auto',
            quantization_config=self.quant_config,
            device_map=get_kbit_device_map(),
        )

        self.policy = PeftModelForCausalLM.from_pretrained(
            base_gen_policy,
            self.args.policy_adapter_path,
            is_trainable=False,
            adapter_name="policy_adapter"
        )

        self.ref_policy = PeftModelForCausalLM.from_pretrained(
            base_gen_ref,
            self.args.ref_adapter_path,
            is_trainable=False,
            adapter_name="ref_adapter"
        )
        
        self.ref_policy.active_adapter = "ref_adapter"
        self.policy.active_adapter = "policy_adapter"

        for name, param in self.policy.named_parameters():
            if self.policy.active_adapter in name:
                param.requires_grad = True

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

        trainable = [n for n, p in self.policy.named_parameters() if p.requires_grad]
        print(f"Trainable policy parameters ({len(trainable)}):")
        for n in trainable:
            print(" -", n)

    def _setup_classification_models(self):
        base_cls_model = AutoModelForSequenceClassification.from_pretrained(
            self.args.model_name,
            torch_dtype='auto',
            num_labels=1,
            quantization_config=self.quant_config,
            device_map=get_kbit_device_map(),
        )

        base_value_model = AutoModelForSequenceClassification.from_pretrained(
            self.args.model_name,
            torch_dtype='auto',
            num_labels=1,
            quantization_config=self.quant_config,
            device_map=get_kbit_device_map(),
        )
        
        self.reward_model = PeftModelForSequenceClassification.from_pretrained(
            base_cls_model,
            self.args.reward_adapter_path,
            is_trainable=False,
            adapter_name="reward_adapter"
        )
        
        self.value_model = PeftModelForSequenceClassification.from_pretrained(
            base_value_model,
            self.args.value_adapter_path,
            is_trainable=False,
            adapter_name="value_adapter"
        )
        
        self.reward_model.active_adapter = "reward_adapter"
        self.value_model.active_adapter = "value_adapter"

        for name, param in self.value_model.named_parameters():
            if self.value_model.active_adapter in name:
                param.requires_grad = True

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

    def _setup_grpo_trainer(self):

        num_processes = self.accelerator.num_processes
        global_batch_size = self.args.per_device_train_batch_size * num_processes

        possible_values = [n for n in range(2, global_batch_size + 1) if global_batch_size % n == 0]
        if not possible_values:
            raise ValueError(f"Global batch size {global_batch_size} is too small. Increase batch size or GPUs.")
        
        num_generations = possible_values[0]
        print(f"Using num_generations = {num_generations} (global_batch_size = {global_batch_size})")

        training_args = GRPOConfig(
            per_device_train_batch_size=self.args.per_device_train_batch_size,
            per_device_eval_batch_size=self.args.per_device_eval_batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            num_train_epochs=self.args.num_train_epochs,
            learning_rate=self.args.learning_rate,
            output_dir=self.args.output_dir,
            save_steps=self.args.save_steps,
            ddp_find_unused_parameters=True,
            num_generations=num_generations
        )

        self.grpo_trainer = GRPOTrainer(
            args=training_args,
            model=self.policy,
            reward_funcs=self.args.reward_adapter_path,
            processing_class=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
        )
        print("GRPOtrainer setup complete")

    def train(self):
        try:
            print("Starting GRPO training...")
            train_stats = self.grpo_trainer.train()
            return train_stats
        except Exception as e:
            print(f"Training error: {str(e)}")
            raise

    


