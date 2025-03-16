import json
from typing import Any, Dict

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class SFTDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizer, max_length: int, template):
        self.data = self.load_sft_data(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template = template

    def load_sft_data(self, file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        rendered_text = self.template.render(
            messages=[
                {"role": "user", "content": item["input"]},
                {"role": "assistant", "content": item["output"]}
            ],
            add_generation_prompt=False
        )

        tokens = self.tokenizer(
            rendered_text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )

        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]

        instruction_text = self.template.render(
            messages=[{"role": "user", "content": item["input"]}],
            add_generation_prompt=True, # important
        )
        instruction_tokens = self.tokenizer(
            instruction_text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )

        labels = input_ids.clone()
        instruction_length = instruction_tokens["input_ids"].size(1)
        labels[:, :instruction_length] = -100

        return {
            "input_ids": input_ids.squeeze(),
            "attention_mask": attention_mask.squeeze(),
            "labels": labels.squeeze(),
        }

    def collate_fn(self, batch):
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [item["input_ids"] for item in batch], batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_masks = torch.nn.utils.rnn.pad_sequence(
            [item["attention_mask"] for item in batch], batch_first=True, padding_value=0
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [item["labels"] for item in batch], batch_first=True, padding_value=-100
        )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_masks,
        }

class RMDataset(Dataset):
    def __init__(self, reward_data_path, tokenizer, template, max_length=512):
        self.data = self.load_reward_data(reward_data_path)
        self.tokenizer = tokenizer
        self.template = template
        self.max_length = max_length

    def load_reward_data(self, file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        reward_value = item["value"]

        rendered_text = self.template.render(
            messages=[
                {"role": "user", "content": item["input"]},
                {"role": "assistant", "content": item["output"]}
            ],
            add_generation_prompt=False
        )

        tokenized_input = self.tokenizer(
            rendered_text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True
        )

        return {
            "input_ids": tokenized_input["input_ids"].squeeze(),
            "attention_mask": tokenized_input["attention_mask"].squeeze(),
            "labels": torch.tensor(reward_value)
        }

class PPODataset(Dataset):
    def __init__(self, reward_data_path, tokenizer, template, max_length=512):
        self.data = self.load_reward_data(reward_data_path)
        self.tokenizer = tokenizer
        self.template = template
        self.max_length = max_length

    def load_reward_data(self, file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Render the conversation using the Jinja template
        input_text = self.template.render(
            messages=[
                {"role": "user", "content": item["instruction"]},
                {"role": "assistant", "content": item["output"]}
            ],
            add_generation_prompt=False,
        )

        # Tokenize with max_length and truncation
        tokenized_input = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            max_length=self.max_length,
            truncation=True
        )

        # Return a dictionary as expected by the Trainer
        return {
            "input_ids": tokenized_input["input_ids"].squeeze(),
            "attention_mask": tokenized_input["attention_mask"].squeeze(),
        }

    def collate_fn(self, batch):
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [item["input_ids"] for item in batch], batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_masks = torch.nn.utils.rnn.pad_sequence(
            [item["attention_mask"] for item in batch], batch_first=True, padding_value=0
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
        }
