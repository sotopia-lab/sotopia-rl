import os

import torch
import torch.distributed as dist
from jinja2 import Environment, FileSystemLoader
from peft import LoraConfig, get_peft_model
from torch.nn import MSELoss
from torch.utils.data import random_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from accelerate import Accelerator

import wandb
from sotopia_rl.data import RMDataset
import torch._dynamo
torch._dynamo.config.suppress_errors = True

os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class SotopiaRMTrainer(Trainer):
    def __init__(self, args, accelerator, **kwargs):
        self.args = args
        self.accelerator = accelerator
        self.device = accelerator.device

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        train_dataset, eval_dataset = self.setup_dataset(tokenizer)

        # Initialize wandb only on the main process
        if self.accelerator.is_main_process:
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

        base_model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=1,
            torch_dtype='auto', 
        )

        model = get_peft_model(base_model, peft_config)
        model.config.pad_token_id = tokenizer.pad_token_id  # important to set the config pad_token_id

        model = self.accelerator.prepare_model(model)

        # Set up the TrainingArguments with DeepSpeed support
        training_args = TrainingArguments(
            output_dir=args.checkpoint_dir,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.val_batch_size,
            num_train_epochs=args.num_epochs,
            logging_steps=1,
            save_steps=args.evaluation_steps,
            eval_steps=args.evaluation_steps,
            logging_dir="./logs",
            gradient_accumulation_steps=args.accumulation_steps,
            learning_rate=args.learning_rate,
            warmup_steps=int(len(train_dataset) * args.warmup_epochs),
            optim="adamw_torch",
            remove_unused_columns=False,
            dataloader_num_workers=4,
            report_to="wandb",
            ddp_find_unused_parameters=False,
        )

        collate_fn = train_dataset.dataset.collate_fn if hasattr(train_dataset, 'dataset') else None

        super().__init__(
            model=model,
            args=training_args,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collate_fn,
            **kwargs
        )
        self.loss_fn = MSELoss()

        if args.checkpoint_path:
            self.load_lora_checkpoint(args.checkpoint_path)

    def setup_dataset(self, tokenizer):
        env = Environment(loader=FileSystemLoader("/".join(self.args.template_path.split("/")[:-1])))
        template = env.get_template(self.args.template_path.split("/")[-1])
        dataset = RMDataset(self.args.reward_data_path, tokenizer, template, self.args.max_length)

        if self.accelerator.is_main_process:
            print(f"dataset: {len(dataset)}")

        train_size = int(len(dataset) * 0.95)
        val_size = len(dataset) - train_size

        # Use deterministic splitter with seed to ensure same split across processes
        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
        return train_dataset, val_dataset

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids = inputs["input_ids"]
        attention_masks = inputs["attention_mask"]
        true_rewards = inputs["labels"]

        outputs = model(input_ids, attention_mask=attention_masks)
        predicted_rewards = outputs.logits.squeeze(-1)  # Shape: (batch_size,)
        loss = self.loss_fn(predicted_rewards, true_rewards)

        return (loss, outputs) if return_outputs else loss

    def save_lora_checkpoint(self, output_dir=None, **kwargs):
        if self.accelerator.is_main_process:
            self.model.save_pretrained(output_dir)

    def load_lora_checkpoint(self, checkpoint_path):
        adapter_model_path = os.path.join(checkpoint_path, 'adapter_model.safetensors')
        peft_config = LoraConfig.from_pretrained(checkpoint_path)

        if os.path.exists(adapter_model_path):
            self.model.load_adapter(checkpoint_path, adapter_name='lora', peft_config=peft_config)
        else:
            if self.accelerator.is_main_process:
                print(f"No adapter model found at {adapter_model_path}.")
                
