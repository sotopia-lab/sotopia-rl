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


os.environ['NCCL_P2P_DISABLE'] = '1'

class SotopiaSFTTrainer:
    def __init__(self, args, accelerator):
        self.args = args
        self.accelerator = accelerator
        self.device = accelerator.device

        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={k: v for k, v in vars(args).items() if isinstance(v, (int, float, str))}
        )

        config = AutoConfig.from_pretrained(args.model_name)
        config.use_cache = False

        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        self.tokenizer.model_max_length = args.max_length

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
            base_model.enable_input_require_grads()
            self.model = get_peft_model(base_model, peft_config)
            if not args.use_qlora:
                self.model = self.model.to(self.device)
        else:
            self.model = base_model

        if args.lora_checkpoint_path:
            self.model = PeftModelForCausalLM.from_pretrained(
                base_model,
                args.lora_checkpoint_path
            )
            print("Loading lora checkpoint from {}".format(args.lora_checkpoint_path))


        self.model = self.accelerator.prepare(self.model)
        self.model = self.accelerator.unwrap_model(self.model)
        self.model = self.model.to(self.device)

        env = Environment(loader=FileSystemLoader(os.path.dirname(args.template_path)))
        self.template = env.get_template(os.path.basename(args.template_path))

        self.setup_dataset()

        training_args = TrainingArguments(
            output_dir=args.checkpoint_dir,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.val_batch_size,
            gradient_accumulation_steps=args.accumulation_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            evaluation_strategy="steps",
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
        )

        self.collate_fn = self.train_dataset.dataset.collate_fn if hasattr(self.train_dataset, 'dataset') else None

        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.collate_fn,
        )
        

    def setup_dataset(self):
        dataset = SFTDataset(self.args.sft_data_path, self.tokenizer, max_length=self.args.max_length, template=self.template)
        train_size = int(0.95 * len(dataset))
        val_size = len(dataset) - train_size
        generator = torch.Generator().manual_seed(42)
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

        print(f"Training dataset size: {len(self.train_dataset)}")
        print(f"Validation dataset size: {len(self.val_dataset)}")

    def train(self):
        self.trainer.train()
        self.save_lora_checkpoint()

    def save_lora_checkpoint(self):
        if self.args.use_lora:
            checkpoint_path = os.path.join(self.args.checkpoint_dir, "best_lora_checkpoint")
            os.makedirs(checkpoint_path, exist_ok=True)
            self.model.save_pretrained(checkpoint_path)
            print(f"LoRA checkpoint saved at {checkpoint_path}")

    def load_lora_checkpoint(self, checkpoint_path):
        self.model = self.model.from_pretrained(self.model, checkpoint_path)
        print(f"LoRA checkpoint loaded from {checkpoint_path}")
