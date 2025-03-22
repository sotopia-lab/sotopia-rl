import os

import numpy as np
import requests
import torch
from jinja2 import Environment, FileSystemLoader
from peft import PeftModelForSequenceClassification
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class RejectionSampler:
    def __init__(
        self,
        sft_model_name,
        sft_model_vllm_api_url,
        reward_model_path,
        reward_model_name,
        template_path,
        max_responses,
        max_length,
        sft_batch_size,
        rm_batch_size,
    ):
        self.max_responses = max_responses
        self.sft_batch_size = sft_batch_size
        self.rm_batch_size = rm_batch_size
        self.sft_model_name = sft_model_name
        self.max_length = max_length
        self.sft_model_vllm_api_url = sft_model_vllm_api_url  # Store vLLM API URL

        self.reward_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
        self.reward_model = self.load_reward_model(reward_model_path)

        env = Environment(loader=FileSystemLoader("/".join(template_path.split("/")[:-1])))
        self.template = env.get_template(template_path.split("/")[-1])

    def load_reward_model(self, reward_model_path):
        """Load reward model with optional QLoRA quantization."""
        print(f"Loading reward model: {reward_model_path}")

        model_kwargs = {
            "torch_dtype": torch.float32, # very important
            "device_map": "auto",
            "num_labels": 1  # For regression task
        }

        base_model = AutoModelForSequenceClassification.from_pretrained(
            reward_model_path,
            **model_kwargs
        )
        base_model.config.pad_token_id = self.tokenizer.pad_token_id

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

    def inference(self, messages, temperature, top_p, max_new_tokens):
        """Generate responses using vLLM API and select the best one based on reward model scores."""
        prompt = self.format_prompt(messages, add_generation_prompt=True)

        total_responses = []
        total_responses_generated = 0

        while total_responses_generated < self.max_responses:
            current_batch_size = min(self.sft_batch_size, self.max_responses - total_responses_generated)

            # Generate responses using vLLM API
            payload = {
                "prompt": prompt,
                "model": self.sft_model_name,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_new_tokens,
                "n": current_batch_size,  # Number of completions to generate
                "stop": [self.tokenizer.eos_token] if self.tokenizer.eos_token else None
            }

            try:
                response = requests.post(self.sft_model_vllm_api_url, json=payload)
                response.raise_for_status()

                api_responses = response.json()
                for completion in api_responses.get("choices", []):
                    if "text" in completion:
                        total_responses.append(completion["text"])

                for response in total_responses[total_responses_generated:]:
                    print(response)

                total_responses_generated += current_batch_size

            except Exception as e:
                print(f"Error calling vLLM API: {e}")
                break

        if not total_responses:
            return "Failed to generate responses from vLLM API."

        messages_list = []
        for response in total_responses:
            messages_with_response = messages + [{'role': 'assistant', 'content': response}]
            messages_list.append(messages_with_response)

        rewards = self.inference_rm(messages_list)
        print(rewards)

        top_index = np.argmax(rewards)
        top_response = total_responses[top_index]

        return top_response if top_response is not None else "No valid responses found."

    def inference_rm(self, messages_list):
        """Score responses using the reward model."""
        rewards = []
        for i in range(0, len(messages_list), self.rm_batch_size):
            batch_messages = messages_list[i:i + self.rm_batch_size]

            prompts = [
                self.format_prompt(msgs, add_generation_prompt=False) for msgs in batch_messages
            ]
            inputs = self.tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length
            ).to(self.reward_device)

            with torch.no_grad():
                outputs = self.reward_model(**inputs, return_dict=True)

            batch_rewards = outputs.logits.squeeze().detach().cpu().numpy()

            if batch_rewards.ndim == 0:
                batch_rewards = [batch_rewards.item()]
            rewards.extend(batch_rewards)

        return rewards
