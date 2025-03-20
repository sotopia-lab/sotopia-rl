import os

import numpy as np
import torch
from jinja2 import Environment, FileSystemLoader
from peft import PeftConfig, PeftModelForSequenceClassification, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

class RejectionSampler:
    def __init__(
        self,
        sft_model_path,
        reward_model_path,
        model_name,
        template_path,
        max_responses,
        max_length=4096,
        sft_batch_size=1,
        rm_batch_size=1,
        use_qlora=False,
    ):
        self.max_responses = max_responses
        self.sft_batch_size = sft_batch_size
        self.rm_batch_size = rm_batch_size
        self.max_length = max_length
        self.use_qlora = use_qlora

        # Set up devices: Assign different devices if multiple GPUs are available
        if torch.cuda.device_count() > 1:
            self.sft_device = torch.device("cuda:0")
            self.reward_device = torch.device("cuda:1")
        else:
            self.sft_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.reward_device = self.sft_device

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load models
        self.sft_model = self.load_sft_model(sft_model_path)
        self.reward_model = self.load_reward_model(reward_model_path)

        # Load Jinja template from file
        env = Environment(loader=FileSystemLoader("/".join(template_path.split("/")[:-1])))
        self.template = env.get_template(template_path.split("/")[-1])

    def get_quantization_config(self):
        """Create and return QLoRA quantization configuration if enabled."""
        if self.use_qlora:
            print("Using QLoRA with 4-bit quantization")
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        return None

    def load_peft_config(self, checkpoint_path):
        """Load PEFT configuration from checkpoint path."""
        try:
            peft_config = PeftConfig.from_pretrained(checkpoint_path)
            return peft_config
        except FileNotFoundError:
            print(f"No PEFT configuration file found in {checkpoint_path}")
            return None

    def load_sft_model(self, model_path):
        """Load SFT model with optional QLoRA quantization."""
        print(f"Loading SFT model: {model_path}")

        quantization_config = self.get_quantization_config()
        model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto" if torch.cuda.device_count() == 1 else "cuda:0",
        }

        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config

        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs
        )

        adapter_path = os.path.join(model_path, 'adapter_model')
        if os.path.exists(adapter_path + '.safetensors') or os.path.exists(adapter_path + '.bin'):
            print(f"Loading adapter from: {model_path}")
            peft_config = self.load_peft_config(model_path)
            if peft_config:
                model = get_peft_model(base_model, peft_config)
                model.load_adapter(model_path, adapter_name="default")
            else:
                model = base_model
        else:
            print(f"No adapter found at {adapter_path}, using base model")
            model = base_model

        model.eval()  # Set to evaluation mode
        return model.to(self.sft_device)

    def load_reward_model(self, reward_model_path):
        """Load reward model with optional QLoRA quantization."""
        print(f"Loading reward model: {reward_model_path}")

        quantization_config = self.get_quantization_config()
        model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto" if torch.cuda.device_count() == 1 else "cuda:1",
            "num_labels": 1  # For regression task
        }

        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config

        base_model = AutoModelForSequenceClassification.from_pretrained(
            reward_model_path,
            **model_kwargs
        )

        # Check and load the adapter if it exists
        adapter_path = os.path.join(reward_model_path, 'adapter_model')
        if os.path.exists(adapter_path + '.safetensors') or os.path.exists(adapter_path + '.bin'):
            print(f"Loading reward adapter from: {reward_model_path}")
            reward_model = PeftModelForSequenceClassification.from_pretrained(base_model, reward_model_path)
        else:
            print(f"No adapter found at {adapter_path}, using base model for reward")
            reward_model = base_model

        reward_model.eval()  # Set to evaluation mode
        return reward_model.to(self.reward_device)

    def format_prompt(self, messages, add_generation_prompt=True):
        """Format the prompt using the template."""
        return self.template.render(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
        )

    def inference(self, messages, temperature=1.0, max_new_tokens=200, stream=False, n=1):
        """Generate responses and select the best one based on reward model scores."""
        prompt = self.format_prompt(messages, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.sft_device)
        prompt_length = inputs['input_ids'].size(1)

        total_responses = []
        total_responses_generated = 0

        while total_responses_generated < self.max_responses:
            current_batch_size = min(self.sft_batch_size, self.max_responses - total_responses_generated)

            # Generate current batch of responses
            with torch.no_grad():
                outputs = self.sft_model.generate(
                    input_ids=inputs['input_ids'].repeat(current_batch_size, 1),
                    attention_mask=inputs['attention_mask'].repeat(current_batch_size, 1),
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    num_return_sequences=1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            # Process generated responses
            for i in range(outputs.size(0)):
                generated_tokens = outputs[i, prompt_length:]  # Slice to keep only new tokens
                response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                total_responses.append(response)

            total_responses_generated += current_batch_size

        # Prepare messages with generated responses
        messages_list = []
        for response in total_responses:
            messages_with_response = messages + [{'role': 'assistant', 'content': response}]
            messages_list.append(messages_with_response)

        # Evaluate responses with reward model in batches
        rewards = self.inference_rm(messages_list)

        # Find the best response
        top_index = np.argmax(rewards)
        top_response = total_responses[top_index]

        return top_response if top_response is not None else "No valid responses found."

    def inference_rm(self, messages_list):
        """Score responses using the reward model."""
        rewards = []
        for i in range(0, len(messages_list), self.rm_batch_size):
            batch_messages = messages_list[i:i + self.rm_batch_size]

            # Format prompts for the current batch
            prompts = [
                self.format_prompt(msgs, add_generation_prompt=False) for msgs in batch_messages
            ]
            inputs = self.tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length
            ).to(self.reward_device)

            # Forward pass through the reward model
            with torch.no_grad():
                outputs = self.reward_model(**inputs, return_dict=True)

            # Extract reward values from the logits
            batch_rewards = outputs.logits.squeeze().detach().cpu().numpy()

            # Handle different shapes (single item vs batch)
            if batch_rewards.ndim == 0:
                batch_rewards = [batch_rewards.item()]
            rewards.extend(batch_rewards)

        return rewards
