import os

import torch
from jinja2 import Environment, FileSystemLoader
from peft import LoraConfig, PeftModelForSequenceClassification
from torch.nn import MSELoss
from torch.utils.data import random_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

import wandb
from sotopia_rl.data import RMDataset

class SotopiaRMTrainer(Trainer):
    def __init__(self, args, **kwargs):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_val_loss = float('inf')
        train_dataset, eval_dataset = self.setup_dataset()

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
        tokenizer.pad_token_id = tokenizer.eos_token_id
        

        base_model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name, 
            num_labels=1,  # For regression task (reward modeling)
            pad_token_id=tokenizer.eos_token_id
        ).to(self.device)
            
        model = PeftModelForSequenceClassification(base_model, peft_config)

        training_args = TrainingArguments(
            output_dir=args.checkpoint_dir,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.val_batch_size,
            num_train_epochs=args.num_epochs,
            evaluation_strategy="steps",
            logging_steps=1,
            save_steps=args.evaluation_steps,
            eval_steps=args.evaluation_steps,
            logging_dir="./logs",
            gradient_accumulation_steps=args.accumulation_steps,
            warmup_steps=int(len(train_dataset) * args.warmup_epochs),
            optim="adamw_torch",
            fp16=False,
            bf16=False,
            remove_unused_columns=False,
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

    def setup_dataset(self):
        env = Environment(loader=FileSystemLoader("/".join(self.args.template_path.split("/")[:-1])))
        template = env.get_template(self.args.template_path.split("/")[-1])
        tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)
        dataset = RMDataset(self.args.reward_data_path, tokenizer, template, self.args.max_length)
        print(f"dataset: {len(dataset)}")
        train_size = int(len(dataset) * 0.95)
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        return train_dataset, val_dataset

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids = inputs["input_ids"].to(self.device)
        attention_masks = inputs["attention_mask"].to(self.device)
        true_rewards = inputs["labels"].to(self.device)

        outputs = model(input_ids, attention_mask=attention_masks)
        predicted_rewards = outputs.logits.squeeze(-1)  # Shape: (batch_size,)
        loss = self.loss_fn(predicted_rewards, true_rewards)
        print(self.model.training)
        print("predicted_rewards", predicted_rewards)
        print("true_rewards", true_rewards)
        return (loss, outputs) if return_outputs else loss

    def save_lora_checkpoint(self, output_dir=None, **kwargs):
        self.model.save_pretrained(output_dir)

    def load_lora_checkpoint(self, checkpoint_path):
        adapter_model_path = os.path.join(checkpoint_path, 'adapter_model.safetensors')
        peft_config = LoraConfig.from_pretrained(checkpoint_path)
        if os.path.exists(adapter_model_path):
            self.model.load_adapter(checkpoint_path, adapter_name='lora', peft_config=peft_config)
        else:
            print(f"No adapter model found at {adapter_model_path}.")
            
