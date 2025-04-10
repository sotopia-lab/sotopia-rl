import os

import torch
import torch.distributed as dist
from jinja2 import Environment, FileSystemLoader
from peft import LoraConfig, get_peft_model
from torch.nn import MSELoss
from torch.utils.data import random_split
from transformers import AutoTokenizer, Trainer, TrainingArguments
from trl import AutoModelForCausalLMWithValueHead  # Updated import

import wandb
from sotopia_rl.data import RMDataset

os.environ['NCCL_P2P_DISABLE'] = '1'

def safe_state_dict(self, *args, **kwargs):
    """Fixed state_dict implementation that avoids OrderedDict mutation"""
    # Get the base model's state dict
    sd = self.pretrained_model.state_dict()
    
    # Safely add the v_head state dict
    v_head_sd = self.v_head.state_dict()
    for k, v in list(v_head_sd.items()):  # Use list() to create a copy for iteration
        sd[f"v_head.{k}"] = v
        
    return sd

# Apply the patch
AutoModelForCausalLMWithValueHead.state_dict = safe_state_dict

class SotopiaRMWithValueHeadTrainer(Trainer):
    def __init__(self, args, **kwargs):
        self.args = args

        # Setup distributed training
        self.setup_distributed()

        train_dataset, eval_dataset = self.setup_dataset()

        # Initialize wandb only on the main process
        if self.is_main_process:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config={k: v for k, v in vars(args).items() if isinstance(v, (int, float, str))}
            )

        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules.split(",")
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

        base_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            args.model_name,
            num_labels=1,
            return_dict=True,
        )
        base_model.config.return_dict = True
        base_model.enable_input_require_grads()  # very important

        if self.args.multi_gpu:
            base_model = base_model.to(self.device)
        else:
            base_model = base_model.to(self.device)

        # Wrap the base model with the LoRA adapter using the generic wrapper
        model = get_peft_model(base_model, peft_config)
        model.config.pad_token_id = tokenizer.pad_token_id  # important to set the config pad_token_id

        # Set up the TrainingArguments with DeepSpeed support
        training_args = TrainingArguments(
            output_dir=args.checkpoint_dir,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.val_batch_size,
            num_train_epochs=args.num_epochs,
            evaluation_strategy="steps",
            logging_steps=1,
            save_steps=args.evaluation_steps,
            eval_steps=1000000,
            logging_dir="./logs",
            gradient_accumulation_steps=args.accumulation_steps,
            learning_rate=args.learning_rate,
            warmup_steps=int(len(train_dataset) * args.warmup_epochs),
            optim="adamw_torch",
            fp16=True,
            remove_unused_columns=False,
            dataloader_num_workers=4,
            # Distributed training settings
            local_rank=self.local_rank,
            # DeepSpeed integration
            deepspeed=args.deepspeed_config if hasattr(args, 'deepspeed_config') and args.deepspeed_config else None,
            ddp_find_unused_parameters=False,
        )

        collate_fn = train_dataset.dataset.collate_fn if hasattr(train_dataset, 'dataset') else None

        super().__init__(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collate_fn,
            **kwargs
        )
        self.loss_fn = MSELoss()

        if args.checkpoint_path:
            self.load_lora_checkpoint(args.checkpoint_path)

    def setup_distributed(self):
        """Set up distributed training environment"""
        # Check if DeepSpeed or distributed training is enabled
        self.args.multi_gpu = torch.cuda.device_count() > 1 and (
            (hasattr(self.args, 'deepspeed_config') and self.args.deepspeed_config is not None) or
            (hasattr(self.args, 'use_distributed') and self.args.use_distributed)
        )

        if self.args.multi_gpu:
            if 'LOCAL_RANK' in os.environ:
                self.local_rank = int(os.environ['LOCAL_RANK'])
            else:
                self.local_rank = self.args.local_rank if hasattr(self.args, 'local_rank') else 0

            if not dist.is_initialized():
                dist.init_process_group(backend='nccl')

            self.world_size = dist.get_world_size()
            self.is_main_process = self.local_rank == 0
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.local_rank)

            if self.is_main_process:
                print(f"Distributed training enabled with {self.world_size} GPUs")
        else:
            self.local_rank = 0
            self.is_main_process = True
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.world_size = 1
            print(f"Training on single {'GPU' if torch.cuda.is_available() else 'CPU'}")

    def setup_dataset(self):
        env = Environment(loader=FileSystemLoader("/".join(self.args.template_path.split("/")[:-1])))
        template = env.get_template(self.args.template_path.split("/")[-1])
        tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)
        dataset = RMDataset(self.args.reward_data_path, tokenizer, template, self.args.max_length)

        if self.is_main_process:
            print(f"dataset: {len(dataset)}")

        train_size = int(len(dataset) * 0.95)
        val_size = len(dataset) - train_size

        # Use deterministic splitter with seed to ensure same split across processes
        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
        return train_dataset, val_dataset

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids = inputs["input_ids"].to(self.device)
        attention_masks = inputs["attention_mask"].to(self.device)
        true_rewards = inputs["labels"].to(self.device)

        outputs = model(input_ids, attention_mask=attention_masks, return_past_key_values=True)
        predicted_rewards = outputs[2].mean(dim=-1).squeeze(-1)
        loss = self.loss_fn(predicted_rewards, true_rewards)

        return (loss, outputs) if return_outputs else loss


    def save_lora_checkpoint(self, output_dir=None, **kwargs):
        # With DeepSpeed, checkpoint saving is handled by the Trainer
        # We only need to ensure LoRA adapter is saved properly
        if self.is_main_process:
            self.model.save_pretrained(output_dir)

    def load_lora_checkpoint(self, checkpoint_path):
        adapter_model_path = os.path.join(checkpoint_path, 'adapter_model.safetensors')
        peft_config = LoraConfig.from_pretrained(checkpoint_path)

        # For DeepSpeed resuming, check if this is a DeepSpeed checkpoint
        if os.path.exists(os.path.join(checkpoint_path, 'zero_pp_rank_0_mp_rank_00_model_states.pt')):
            if self.is_main_process:
                print(f"DeepSpeed checkpoint detected at {checkpoint_path}")
            return

        if os.path.exists(adapter_model_path):
            self.model.load_adapter(checkpoint_path, adapter_name='lora', peft_config=peft_config)
        else:
            if self.is_main_process:
                print(f"No adapter model found at {adapter_model_path}.")
