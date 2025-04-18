import os

import torch
from jinja2 import Environment, FileSystemLoader
from peft import LoraConfig, get_peft_model, PeftModelForCausalLM
from torch.utils.data import random_split
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer
import wandb
from sotopia_rl.data import SFTDataset
from datasets import Dataset

os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class SotopiaSFTTrainer:
    def __init__(self, args, accelerator):
        self.args = args
        self.accelerator = accelerator
        self.device = accelerator.device

        if self.accelerator.is_main_process:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config={k: v for k, v in vars(args).items() if isinstance(v, (int, float, str))}
            )

        config = AutoConfig.from_pretrained(args.model_name)
        config.use_cache = False

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        tokenizer.model_max_length = args.max_length

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        if args.use_qlora:
            print(f"Using QLoRA (4bit) to load model: {args.model_name}")
            base_model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                torch_dtype=torch.float16,
                quantization_config=quantization_config,
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(args.model_name).to(self.device)
        
        if args.use_lora:
            peft_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=args.target_modules.split(",")
            )
            self.model = get_peft_model(base_model, peft_config)
            if not args.use_qlora:
                self.model = self.model.to(self.device)
        else:
            self.model = base_model

        #if args.lora_checkpoint_path:
        #    self.load_lora_checkpoint(args.lora_checkpoint_path)

        self.model = self.accelerator.prepare(self.model)
        self.model = self.accelerator.unwrap_model(self.model)
        self.model = self.model.to(self.device)

        env = Environment(loader=FileSystemLoader(os.path.dirname(args.template_path)))
        self.template = env.get_template(os.path.basename(args.template_path))

        train_dataset, val_dataset = self.setup_dataset(tokenizer)

        training_args = TrainingArguments(
            output_dir=args.checkpoint_dir,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.val_batch_size,
            gradient_accumulation_steps=args.accumulation_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            eval_steps=args.evaluation_steps,
            save_steps=args.evaluation_steps,
            logging_dir="./logs",
            logging_steps=1,
            report_to="wandb",
            gradient_checkpointing=False,
            optim="paged_adamw_8bit" if args.use_qlora else "adamw_torch",
            bf16=True,
            fp16=False,
            dataloader_num_workers=4,
            ddp_find_unused_parameters=False,
        )

        collate_fn = train_dataset.dataset.collate_fn if hasattr(train_dataset, 'dataset') else None

        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=tokenizer,
            data_collator=collate_fn,
        )
        

    def setup_dataset(self, tokenizer):
        env = Environment(loader=FileSystemLoader("/".join(self.args.template_path.split("/")[:-1])))
        template = env.get_template(self.args.template_path.split("/")[-1])
        dataset = SFTDataset(self.args.sft_data_path, tokenizer, template, self.args.max_length)

        if self.accelerator.is_main_process:
            print(f"dataset: {len(dataset)}")

        train_size = int(len(dataset) * 0.95)
        val_size = len(dataset) - train_size

        # Use deterministic splitter with seed to ensure same split across processes
        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
        train_dataset = Dataset.from_list([train_dataset[i] for i in range(len(train_dataset))])
        val_dataset = Dataset.from_list([val_dataset[i] for i in range(len(val_dataset))])

        return train_dataset, val_dataset

    def train(self):
        self.trainer.train()
        self.save_lora_checkpoint()

    def save_lora_checkpoint(self):
        if self.args.use_lora:
            checkpoint_path = os.path.join(self.args.checkpoint_dir, "best_lora_checkpoint")
            os.makedirs(checkpoint_path, exist_ok=True)
            self.model.save_pretrained(checkpoint_path)
            print(f"LoRA checkpoint saved at {checkpoint_path}")

    def load_lora_checkpoint(self, checkpoint_path, is_trainable=True):
        self.model.load_adapter(self.model, checkpoint_path, is_trainable=is_trainable) # important to set is_trainable
        print(f"LoRA checkpoint loaded from {checkpoint_path}")