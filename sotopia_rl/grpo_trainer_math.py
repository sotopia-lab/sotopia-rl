import os
import torch
import wandb
from datasets import load_dataset
from torch.utils.data import random_split
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)
from accelerate import PartialState
from peft import PeftModelForCausalLM, PeftModelForSequenceClassification
from jinja2 import Environment, FileSystemLoader
from trl import get_kbit_device_map, GRPOConfig, GRPOTrainer
from accelerate import Accelerator
from sotopia_rl.data import GRPODataset
from functools import partial
from typing import List

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

SIMPLE_CHAT_TEMPLATE = "{% for message in messages %}{{message['role'].capitalize() + ': ' + message['content'] + '\n\n'}}{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"


from transformers import GPTNeoXForCausalLM

class PatchedGPTNeoXForCausalLM(GPTNeoXForCausalLM):
    def forward(self, *args, logits_to_keep=None, **kwargs):
        return super().forward(*args, **kwargs)

class SotopiaGRPOTrainer:
    def __init__(self, args, accelerator: Accelerator):
        self.args = args
        self.accelerator = accelerator

        self._init_wandb()
        self._setup_tokenizer()
        self._setup_dataset()
        self._create_quantization_config()

        self._setup_grpo_trainer()

        def save_model(self, output_dir: str, _internal_call: bool = False):
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            print(f"Saved PEFT model to {output_dir}")

        self.grpo_trainer.save_model = save_model.__get__(self.grpo_trainer, type(self.grpo_trainer))

    def _init_wandb(self):
        wandb.init(
            project=self.args.wandb_project,
            name=self.args.wandb_run_name,
            config={k: v for k, v in vars(self.args).items() if isinstance(v, (int, float, str))}
        )

    def _setup_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained("/data/disk0/models/EleutherAI_pythia-1b-deduped__sft__tldr")
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids('[PAD]')
        if self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE


    def _setup_dataset(self):
        from datasets import load_dataset

        dataset = load_dataset("trl-internal-testing/tldr-preference-sft-trl-style")
        print("processing")
        train_dataset = dataset["train"]
        eval_dataset = dataset["validation"]

        def prepare_dataset(dataset, tokenizer):
            def tokenize(element):
                input_ids = tokenizer.apply_chat_template(
                    element["messages"][:1],
                    padding=False,
                    add_generation_prompt=True,
                )
                return {"input_ids": input_ids, "lengths": len(input_ids), "prompt": element["messages"][:1]}

            return dataset.map(
                tokenize,
                remove_columns=dataset.column_names,
                num_proc=4,
            )

        with PartialState().local_main_process_first():
            train_dataset = prepare_dataset(train_dataset, self.tokenizer)
            if eval_dataset is not None:
                eval_dataset = prepare_dataset(eval_dataset, self.tokenizer)
            train_dataset = train_dataset.filter(lambda x: x["lengths"] <= 512, num_proc=4)
            if eval_dataset is not None:
                eval_dataset = eval_dataset.filter(lambda x: x["lengths"] <= 512, num_proc=4)

        assert train_dataset[0]["input_ids"][-1] != self.tokenizer.eos_token_id, "The last token should not be an EOS token"

        self.train_dataset = train_dataset
        self.val_dataset = eval_dataset
        print(f"Dataset loaded and processed: {len(self.train_dataset)} train, {len(self.val_dataset or [])} validation")

    def _create_quantization_config(self):
        self.quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

    def _setup_grpo_trainer(self):
        num_processes = self.accelerator.num_processes
        global_batch_size = self.args.per_device_train_batch_size * num_processes

        num_generations = 4  # manually chosen value
        print(f"Using num_generations = {num_generations} (global_batch_size = {global_batch_size})")

        policy_model = AutoModelForCausalLM.from_pretrained(
                "/data/disk0/models/EleutherAI_pythia-1b-deduped__sft__tldr",
                torch_dtype='auto',
                num_labels=1,
            )
        
        reward_model = AutoModelForSequenceClassification.from_pretrained(
                "/data/disk0/models/EleutherAI_pythia-1b-deduped__reward__tldr",
                torch_dtype='auto',
                num_labels=1,
            )

        training_args = GRPOConfig(
            logging_steps = 1,
            report_to = "wandb",
            per_device_train_batch_size=self.args.per_device_train_batch_size,
            per_device_eval_batch_size=self.args.per_device_eval_batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            num_train_epochs=self.args.num_train_epochs,
            learning_rate=self.args.learning_rate,
            output_dir=self.args.output_dir,
            save_steps=self.args.save_steps,
            num_generations=num_generations
        )

        self.grpo_trainer = GRPOTrainer(
            args=training_args,
            model=policy_model,
            reward_funcs=reward_model,
            processing_class=self.tokenizer,
            reward_processing_classes=self.tokenizer,
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
