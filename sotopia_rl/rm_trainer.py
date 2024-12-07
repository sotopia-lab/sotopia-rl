import os

import torch
import wandb
from jinja2 import Environment, FileSystemLoader
from peft import LoraConfig, PeftModelForCausalLM
from torch.nn import MSELoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import random_split
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from trl import AutoModelForCausalLMWithValueHead

from .data import RMDataset


class SotopiaRMTrainer(Trainer):
    def __init__(self, args, **kwargs):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_val_loss = float('inf')  # Initialize best validation loss
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
        model = AutoModelForCausalLM.from_pretrained(args.model_name)
        model = PeftModelForCausalLM(model, peft_config)
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

        optimizer = self.create_custom_optimzer(model, args)
        lr_scheduler = self.create_custom_scheduler(optimizer, len(train_dataset) // args.train_batch_size)

        training_args = TrainingArguments(
            output_dir=args.checkpoint_dir,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.val_batch_size,
            num_train_epochs=args.num_epochs,
            evaluation_strategy="steps",
            logging_steps=50,
            save_steps=args.evaluation_steps,
            eval_steps=args.evaluation_steps,
            logging_dir="./logs",
            gradient_accumulation_steps=args.accumulation_steps,
            warmup_steps=int(len(train_dataset) * args.warmup_epochs),
            load_best_model_at_end=True,
            save_total_limit=10,  # Only keep the latest 3 checkpoints
            label_names=["labels"],
            save_safetensors=False
        )


        super().__init__(
            model=model,
            args=training_args,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            optimizers=(optimizer, lr_scheduler),
            data_collator=self.collate_fn,
            **kwargs
        )
        self.loss_fn = MSELoss()

        if args.checkpoint_path:
            self.load_checkpoint(args.checkpoint_path)


    def collate_fn(self, batch):
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [item["input_ids"] for item in batch], batch_first=True, padding_value=self.processing_class.pad_token_id
        )
        attention_masks = torch.nn.utils.rnn.pad_sequence(
            [item["attention_mask"] for item in batch], batch_first=True, padding_value=0
        )
        labels = torch.stack([item["labels"] for item in batch])  # Stack labels into a tensor

        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "labels": labels,  # Ensure labels are present in the batch output
        }


    def setup_dataset(self):
        # Load dataset and create train/val split
        env = Environment(loader=FileSystemLoader("/".join(self.args.template_path.split("/")[:-1])))
        template = env.get_template(self.args.template_path.split("/")[-1])
        tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)
        dataset = RMDataset(self.args.reward_data_path, tokenizer, template, self.args.max_length)

        #train_size = int(len(dataset) * 0.95)
        train_size = 40
        #val_size = len(dataset) - train_size
        val_size = 10
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        return train_dataset, val_dataset


    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Custom compute_loss for value head-based reward model
        input_ids = inputs["input_ids"].to(self.device)
        attention_masks = inputs["attention_mask"].to(self.device)
        true_rewards = inputs["labels"].to(self.device)

        # Forward pass
        _, _, outputs = model(input_ids, attention_mask=attention_masks, return_dict=True)

        # Compute loss using value head outputs
        last_indices = (attention_masks.sum(dim=1) - 1).long()
        eos_values = outputs[torch.arange(outputs.size(0)), last_indices]
        loss = self.loss_fn(eos_values, true_rewards)

        return (loss, outputs) if return_outputs else loss


    def create_custom_optimzer(self, model, args):
        return AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    def create_custom_scheduler(self, optimizer, steps_per_epoch, **kwargs):
        warmup_scheduler = LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=int(steps_per_epoch * self.args.warmup_epochs)
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=self.args.num_epochs * steps_per_epoch, eta_min=self.args.min_lr
        )
        return SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[int(steps_per_epoch * self.args.warmup_epochs)])

    def save_model(self, output_dir=None, **kwargs):
        state_dict = self.model.state_dict()
        decoder_state_dict = {}
        v_head_state_dict = {}
        for name, param in state_dict.items():
            if name.startswith("v_head."):
                v_head_state_dict[name.replace("v_head.", "")] = param
            else:
                decoder_state_dict[name.replace("pretrained_model.", "")] = param

        self.model.pretrained_model.save_pretrained(
            output_dir,
            state_dict=decoder_state_dict or None,
        )
        torch.save(v_head_state_dict, os.path.join(output_dir, 'value_head.pt'))


    def _load_best_model(self):
        checkpoint_path = self.state.best_model_checkpoint
        self.load_checkpoint(checkpoint_path)


    def load_checkpoint(self, checkpoint_path):
        adapter_model_path = os.path.join(checkpoint_path, 'adapter_model.safetensors')
        if os.path.exists(adapter_model_path):
            self.model.pretrained_model.load_adapter(checkpoint_path, adapter_name='lora')
        else:
            print(f"No adapter model found at {adapter_model_path}.")

        value_head_path = os.path.join(checkpoint_path, 'value_head.pt')
        if os.path.exists(value_head_path):
            value_head_state_dict = torch.load(value_head_path, map_location=self.device)
            self.model.v_head.load_state_dict(value_head_state_dict, strict=True)
        else:
            print(f"No value head state found at {value_head_path}.")

        optimizer_path = os.path.join(checkpoint_path, 'optimizer.pt')
        if os.path.exists(optimizer_path):
            self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))
        else:
            print(f"No optimizer state found at {optimizer_path}.")

        scheduler_path = os.path.join(checkpoint_path, 'scheduler.pt')
        if os.path.exists(scheduler_path):
            self.lr_scheduler.load_state_dict(torch.load(scheduler_path, map_location=self.device, weights_only=True))
        else:
            print(f"No scheduler state found at {scheduler_path}.")
