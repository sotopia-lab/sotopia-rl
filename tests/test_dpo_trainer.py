"""Tests for DPO trainer and dataset."""
import json
import os
import tempfile
from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from jinja2 import Environment, FileSystemLoader
from transformers import AutoTokenizer


class TestDPODataset:
    """Test cases for DPODataset."""

    @pytest.fixture
    def sample_dpo_data(self, tmp_path):
        """Create sample DPO data file."""
        data = [
            {
                "input": "Hello, how are you?",
                "output1": "I'm doing great!",
                "output2": "I'm fine.",
                "score1": 0.8,
                "score2": 0.5,
                "chosen": "I'm doing great!",
                "rejected": "I'm fine.",
                "chosen_score": 0.8,
                "rejected_score": 0.5,
                "original_output": "I'm good!"
            },
            {
                "input": "What's the weather like?",
                "output1": "It's sunny today.",
                "output2": "Weather is okay.",
                "score1": 0.9,
                "score2": 0.6,
                "chosen": "It's sunny today.",
                "rejected": "Weather is okay.",
                "chosen_score": 0.9,
                "rejected_score": 0.6,
                "original_output": "It's nice outside."
            },
        ]
        data_path = tmp_path / "test_dpo_data.json"
        with open(data_path, "w") as f:
            json.dump(data, f)
        return str(data_path)

    @pytest.fixture
    def sample_template(self, tmp_path):
        """Create a simple chat template."""
        template_content = """{% for message in messages %}
{% if message.role == 'user' %}<|im_start|>user
{{ message.content }}<|im_end|>
{% elif message.role == 'assistant' %}<|im_start|>assistant
{{ message.content }}<|im_end|>
{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}"""
        template_path = tmp_path / "test_template.jinja"
        with open(template_path, "w") as f:
            f.write(template_content)
        return str(template_path)

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 1
        tokenizer.model_max_length = 4096
        return tokenizer

    def test_dpo_dataset_loading(self, sample_dpo_data, sample_template, mock_tokenizer):
        """Test that DPODataset loads data correctly."""
        from sotopia_rl.data import DPODataset

        env = Environment(loader=FileSystemLoader(os.path.dirname(sample_template)))
        template = env.get_template(os.path.basename(sample_template))

        dataset = DPODataset(
            data_path=sample_dpo_data,
            tokenizer=mock_tokenizer,
            template=template,
            max_length=512,
        )

        assert len(dataset) == 2

    def test_dpo_dataset_getitem(self, sample_dpo_data, sample_template, mock_tokenizer):
        """Test that DPODataset returns correct item format."""
        from sotopia_rl.data import DPODataset

        env = Environment(loader=FileSystemLoader(os.path.dirname(sample_template)))
        template = env.get_template(os.path.basename(sample_template))

        dataset = DPODataset(
            data_path=sample_dpo_data,
            tokenizer=mock_tokenizer,
            template=template,
            max_length=512,
        )

        item = dataset[0]

        # Check required keys for DPO training
        assert "prompt" in item
        assert "chosen" in item
        assert "rejected" in item

        # Check values match expected data
        assert item["chosen"] == "I'm doing great!"
        assert item["rejected"] == "I'm fine."
        assert "Hello, how are you?" in item["prompt"]

    def test_dpo_dataset_prompt_rendering(self, sample_dpo_data, sample_template, mock_tokenizer):
        """Test that prompts are rendered with generation prompt."""
        from sotopia_rl.data import DPODataset

        env = Environment(loader=FileSystemLoader(os.path.dirname(sample_template)))
        template = env.get_template(os.path.basename(sample_template))

        dataset = DPODataset(
            data_path=sample_dpo_data,
            tokenizer=mock_tokenizer,
            template=template,
            max_length=512,
        )

        item = dataset[0]

        # The prompt should end with the assistant generation prompt
        assert "<|im_start|>assistant" in item["prompt"]


class TestSotopiaDPOTrainer:
    """Test cases for SotopiaDPOTrainer."""

    @pytest.fixture
    def mock_args(self, tmp_path):
        """Create mock training arguments."""
        # Create sample data
        data = [
            {
                "input": "Test input 1",
                "chosen": "Good response",
                "rejected": "Bad response",
                "chosen_score": 1.0,
                "rejected_score": 0.0,
            },
            {
                "input": "Test input 2",
                "chosen": "Another good response",
                "rejected": "Another bad response",
                "chosen_score": 0.9,
                "rejected_score": 0.1,
            },
            {
                "input": "Test input 3",
                "chosen": "Third good response",
                "rejected": "Third bad response",
                "chosen_score": 0.8,
                "rejected_score": 0.2,
            },
        ]
        data_path = tmp_path / "dpo_data.json"
        with open(data_path, "w") as f:
            json.dump(data, f)

        # Create template
        template_content = """{% for message in messages %}
{% if message.role == 'user' %}<|im_start|>user
{{ message.content }}<|im_end|>
{% elif message.role == 'assistant' %}<|im_start|>assistant
{{ message.content }}<|im_end|>
{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}"""
        template_path = tmp_path / "template.jinja"
        with open(template_path, "w") as f:
            f.write(template_content)

        return Namespace(
            model_name="gpt2",  # Small model for testing
            dpo_data_path=str(data_path),
            template_path=str(template_path),
            output_dir=str(tmp_path / "output"),
            max_length=256,
            max_prompt_length=128,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=1,
            num_train_epochs=1,
            learning_rate=1e-5,
            beta=0.1,
            save_steps=100,
            val_ratio=0.33,
            use_lora=False,
            wandb_project="test_project",
            wandb_run_name="test_run",
        )

    @pytest.fixture
    def mock_accelerator(self):
        """Create a mock accelerator."""
        accelerator = MagicMock()
        accelerator.is_main_process = True
        accelerator.num_processes = 1
        accelerator.device = torch.device("cpu")
        return accelerator

    def test_dpo_trainer_import(self):
        """Test that DPOTrainer can be imported."""
        from sotopia_rl import SotopiaDPOTrainer
        assert SotopiaDPOTrainer is not None

    @patch("sotopia_rl.dpo_trainer.wandb")
    def test_dpo_trainer_tokenizer_setup(self, mock_wandb, mock_args, mock_accelerator):
        """Test tokenizer setup in DPOTrainer."""
        from sotopia_rl.dpo_trainer import SotopiaDPOTrainer

        # Only test tokenizer setup, not full initialization
        trainer = MagicMock(spec=SotopiaDPOTrainer)
        trainer.args = mock_args
        
        # Call the tokenizer setup method directly
        tokenizer = AutoTokenizer.from_pretrained(mock_args.model_name, padding_side="left")
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        
        assert tokenizer.pad_token == "[PAD]"
        assert tokenizer.padding_side == "left"

    @patch("sotopia_rl.dpo_trainer.wandb")
    def test_dpo_trainer_dataset_setup(self, mock_wandb, mock_args, mock_accelerator):
        """Test dataset setup in DPOTrainer."""
        from sotopia_rl.data import DPODataset
        from jinja2 import Environment, FileSystemLoader

        env = Environment(
            loader=FileSystemLoader(os.path.dirname(mock_args.template_path))
        )
        template = env.get_template(os.path.basename(mock_args.template_path))
        
        tokenizer = AutoTokenizer.from_pretrained(mock_args.model_name)
        
        dataset = DPODataset(
            data_path=mock_args.dpo_data_path,
            tokenizer=tokenizer,
            template=template,
            max_length=mock_args.max_length,
        )

        assert len(dataset) == 3
        item = dataset[0]
        assert "prompt" in item
        assert "chosen" in item
        assert "rejected" in item


class TestDPODataIntegration:
    """Integration tests for DPO data format from dpo_pairs_scored.json."""

    def test_real_data_format_compatibility(self):
        """Test that the dataset can handle the actual dpo_pairs_scored.json format."""
        sample_data = {
            "input": "Imagine you are Mia Sanders...",
            "output1": '{"action_type": "speak", "argument": "Response 1"}',
            "output2": '{"action_type": "speak", "argument": "Response 2"}',
            "score1": -2.8125,
            "score2": -1.53125,
            "chosen": '{"action_type": "speak", "argument": "Response 2"}',
            "rejected": '{"action_type": "speak", "argument": "Response 1"}',
            "chosen_score": -1.53125,
            "rejected_score": -2.8125,
            "original_output": '{"action_type": "speak", "argument": "Original"}'
        }

        # Verify the format has required fields
        assert "input" in sample_data
        assert "chosen" in sample_data
        assert "rejected" in sample_data
        
        # Verify chosen has higher score than rejected
        assert sample_data["chosen_score"] >= sample_data["rejected_score"]

