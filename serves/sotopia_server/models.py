import os
os.environ["REDIS_OM_URL"] = "redis://:QzmCUD3C3RdsR@34.132.61.229:6379"

import numpy as np
import torch
from jinja2 import Environment, FileSystemLoader
from peft import PeftConfig, PeftModelForCausalLM, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead

from sotopia.database import EpisodeLog
from sotopia_rl.ppo_trainer import SotopiaPPOTrainer

class RejectionSampler:
    def __init__(self,
        sft_model_path,
        reward_model_path,
        model_name,
        template_path,
        max_responses,
        max_length=4096,
        sft_batch_size=1,
        rm_batch_size=1,
    ):
        self.max_responses = max_responses
        self.sft_batch_size = sft_batch_size
        self.rm_batch_size = rm_batch_size

        # Set up devices: Assign different devices if multiple GPUs are available
        if torch.cuda.device_count() > 1:
            self.sft_device = torch.device("cuda:0")
            self.reward_device = torch.device("cuda:1")
        else:
            self.sft_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.reward_device = self.sft_device

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        sft_peft_config = self.load_peft_config(sft_model_path)
        rm_peft_config = self.load_peft_config(reward_model_path)

        # Load SFT model and move it to its designated device
        model = AutoModelForCausalLM.from_pretrained(sft_model_path)
        sft_model = get_peft_model(model, sft_peft_config)
        self.sft_model = self.load_sft_model(sft_model, sft_model_path)

        # Load reward model and move it to its designated device
        reward_model = AutoModelForCausalLM.from_pretrained(model_name)
        reward_model = PeftModelForCausalLM(reward_model, rm_peft_config)
        reward_model = AutoModelForCausalLMWithValueHead.from_pretrained(reward_model)
        self.reward_model = self.load_reward_model(reward_model, reward_model_path)

        # Load Jinja template from file
        env = Environment(loader=FileSystemLoader("/".join(template_path.split("/")[:-1])))
        self.template = env.get_template(template_path.split("/")[-1])
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def load_peft_config(self, checkpoint_path):
        try:
            peft_config = PeftConfig.from_pretrained(checkpoint_path)
            return peft_config
        except FileNotFoundError:
            print(f"No PEFT configuration file found in {checkpoint_path}")
            return None

    def load_sft_model(self, model, checkpoint_path):
        model.load_adapter(checkpoint_path, adapter_name="default")
        return model.to(self.sft_device)

    def load_reward_model(self, model, checkpoint_path):
        adapter_model_path = os.path.join(checkpoint_path, 'adapter_model.safetensors')
        if os.path.exists(adapter_model_path):
            model.pretrained_model.load_adapter(checkpoint_path, adapter_name='lora')
        else:
            print(f"No adapter model found at {adapter_model_path}.")

        value_head_path = os.path.join(checkpoint_path, 'value_head.pt')
        if os.path.exists(value_head_path):
            value_head_state_dict = torch.load(value_head_path, map_location=self.reward_device, weights_only=True)
            new_value_head_state_dict = {}
            for name, param in value_head_state_dict.items():
                if name.startswith('v_head.'):
                    new_value_head_state_dict[name[len('v_head.'):]] = value_head_state_dict[name]
            model.v_head.load_state_dict(new_value_head_state_dict, strict=True)
        else:
            print(f"No value head state found at {value_head_path}.")

        return model.to(self.reward_device)


    def format_prompt(self, messages, add_generation_prompt=True):
        return self.template.render(
            bos_token=self.tokenizer.bos_token,
            messages=messages,
            add_generation_prompt=add_generation_prompt,
        )

    def inference(self, messages, temperature=1.0, stream=False, n=1):
        prompt = self.format_prompt(messages, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.sft_device)
        prompt_length = inputs['input_ids'].size(1)

        total_responses = []
        total_responses_generated = 0

        while total_responses_generated < self.max_responses:
            current_batch_size = min(self.sft_batch_size, self.max_responses - total_responses_generated)

            # Generate current batch of responses
            outputs = self.sft_model.generate(
                input_ids=inputs['input_ids'].repeat(current_batch_size, 1),
                attention_mask=inputs['attention_mask'].repeat(current_batch_size, 1),
                max_new_tokens=200,
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
            _, _, outputs = self.reward_model(**inputs, return_dict=True)

            # Extract reward values
            attention_masks = inputs['attention_mask']
            last_indices = (attention_masks.sum(dim=1) - 1).long()
            eos_values = outputs[torch.arange(outputs.size(0)), last_indices]

            batch_rewards = eos_values.detach().cpu().numpy()
            rewards.extend(batch_rewards)

        return rewards

class OnlinePPORejectionSampler(RejectionSampler, SotopiaPPOTrainer):
    def __init__(self,
        sft_model_path,
        reward_model_path,
        model_name,
        template_path,
        max_responses,
        max_length=4096,
        sft_batch_size=1,
        rm_batch_size=1,
    ):
        RejectionSampler.__init__(self,        
                                    sft_model_path,
                                    reward_model_path,
                                    model_name,
                                    template_path,
                                    max_responses,
                                    max_length=max_length,
                                    sft_batch_size=sft_batch_size,
                                    rm_batch_size=rm_batch_size)
        # for ppo trainer
        self.train_batch_size = sft_batch_size
        self.policy_model = self.sft_model
        self.device = self.sft_device
        self.dataset = None
    
    def train_on_episode_tag(self, tag: str):
        Episodes = EpisodeLog.find(EpisodeLog.tag == "sotopia_rejection-sampling-rm-direct-prompt-and-sft_vs_sotopia_gemma-2-2b-it-sft-1204_sample_40").all()
        print(f"Training on episodes: {Episodes}")
        