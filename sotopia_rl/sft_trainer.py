import os

import torch
import torch.distributed as dist
from jinja2 import Environment, FileSystemLoader
from peft import LoraConfig, get_peft_model
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


class SotopiaSFTTrainer:
    def __init__(self, args):
        self.args = args

        # Setup distributed training
        self.setup_distributed()

        # Initialize wandb only on the main process
        if self.is_main_process:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config={k: v for k, v in vars(args).items() if isinstance(v, (int, float, str))}
            )

        config = AutoConfig.from_pretrained(args.model_name)
        config.use_cache = False

        # Load the tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
        self.tokenizer.model_max_length = args.max_length

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        if args.use_qlora:
            if self.is_main_process:
                print(f"Using QLoRA (4bit) to load model: {args.model_name}")
            base_model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                torch_dtype=torch.float16,
                device_map="auto" if not self.args.multi_gpu else {"": self.local_rank},
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
            base_model.enable_input_require_grads() # very important
            self.model = get_peft_model(base_model, peft_config)
            if not args.use_qlora:
                self.model = self.model.to(self.device)
        else:
            self.model = base_model

        # Load LoRA checkpoint if specified
        if args.lora_checkpoint_path:
            self.load_lora_checkpoint(args.lora_checkpoint_path)

        env = Environment(loader=FileSystemLoader("/".join(self.args.template_path.split("/")[:-1])))
        self.template = env.get_template(self.args.template_path.split("/")[-1])

        # Set up dataset and data loaders
        self.setup_dataset()

        # Set up the SFT Trainer with appropriate arguments
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
            gradient_checkpointing=True if self.args.multi_gpu else False,
            optim="paged_adamw_8bit" if args.use_qlora else "adamw_torch",
            fp16=True,
            # Distributed training settings
            local_rank=self.local_rank,
            ddp_find_unused_parameters=False,
            # Added for multi-GPU support
            dataloader_num_workers=4,
            deepspeed=args.deepspeed_config if hasattr(args, 'deepspeed_config') else None,
        )

        self.collate_fn = self.train_dataset.dataset.collate_fn if hasattr(self.train_dataset, 'dataset') else None

        # Initialize SFTTrainer
        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.collate_fn,
        )

    def setup_distributed(self):
        """Set up distributed training environment"""
        self.args.multi_gpu = torch.cuda.device_count() > 1 and self.args.use_distributed

        if self.args.multi_gpu:
            # Initialize the distributed environment
            if 'LOCAL_RANK' in os.environ:
                self.local_rank = int(os.environ['LOCAL_RANK'])
            else:
                self.local_rank = self.args.local_rank if hasattr(self.args, 'local_rank') else 0

            # Initialize the process group
            if not dist.is_initialized():
                dist.init_process_group(backend='nccl')

            self.world_size = dist.get_world_size()
            self.is_main_process = self.local_rank == 0
            self.device = torch.device(f"cuda:{self.local_rank}")

            # Set the device for this process
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
        # Load dataset and create train/val split
        dataset = SFTDataset(self.args.sft_data_path, self.tokenizer, max_length=self.args.max_length,
                             template=self.template)
        train_size = int(0.95 * len(dataset))
        val_size = len(dataset) - train_size

        # Use deterministic splitter with seed to ensure same split across processes
        generator = torch.Generator().manual_seed(42)
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

        if self.is_main_process:
            print(f"Training dataset size: {len(self.train_dataset)}")
            print(f"Validation dataset size: {len(self.val_dataset)}")

    def train(self):
        # Begin training with SFTTrainer
        self.trainer.train()

        # Save the final model (only on main process)
        if self.is_main_process:
            self.save_lora_checkpoint()

    def save_lora_checkpoint(self):
        # Save model in standard PEFT format if LoRA is used
        if self.args.use_lora:
            checkpoint_path = os.path.join(self.args.checkpoint_dir, "best_lora_checkpoint")
            os.makedirs(checkpoint_path, exist_ok=True)
            self.model.save_pretrained(checkpoint_path)
            print(f"LoRA checkpoint saved at {checkpoint_path}")

    def load_lora_checkpoint(self, checkpoint_path):
        # Load LoRA checkpoint in standard PEFT format
        self.model = self.model.from_pretrained(self.model, checkpoint_path)
        print(f"LoRA checkpoint loaded from {checkpoint_path}")
