import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead
from jinja2 import Template
from jinja2 import Environment, FileSystemLoader
from peft import PeftConfig, get_peft_model, PeftModelForCausalLM
import os

class RejectionSampler:
    def __init__(self, sft_model_path, reward_model_path, model_name, template_path, rejection_threshold=0.5, max_samples=5, max_length=4096):
        self.rejection_threshold = rejection_threshold
        self.max_samples = max_samples

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
        model = AutoModelForCausalLM.from_pretrained(sft_model_path).to(self.sft_device)
        self.sft_model = get_peft_model(model, sft_peft_config)
        self.load_sft_model(sft_model_path)
        

        # Load reward model and move it to its designated device
        reward_model = AutoModelForCausalLM.from_pretrained(model_name)
        reward_model = PeftModelForCausalLM(reward_model, rm_peft_config)
        self.reward_model = AutoModelForCausalLMWithValueHead.from_pretrained(reward_model)
        self.load_reward_model(reward_model_path)

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

    def load_sft_model(self, checkpoint_path):
        self.sft_model.load_adapter(checkpoint_path, adapter_name="default")
        self.sft_model = self.sft_model.to(self.sft_device)

    def load_reward_model(self, checkpoint_path):
        adapter_model_path = os.path.join(checkpoint_path, 'adapter_model.safetensors')
        if os.path.exists(adapter_model_path):
            self.reward_model.pretrained_model.load_adapter(checkpoint_path, adapter_name='lora')
        else:
            print(f"No adapter model found at {adapter_model_path}.")

        value_head_path = os.path.join(checkpoint_path, 'value_head.pt')
        if os.path.exists(value_head_path):
            value_head_state_dict = torch.load(value_head_path, map_location=self.reward_device)
            new_value_head_state_dict = {}
            for name, param in value_head_state_dict.items():
                if name.startswith('v_head.'):
                    new_value_head_state_dict[name[len('v_head.'):]] = value_head_state_dict[name]
            self.reward_model.v_head.load_state_dict(new_value_head_state_dict, strict=True)
        else:
            print(f"No value head state found at {value_head_path}.")

        self.reward_model = self.reward_model.to(self.reward_device)


    def format_prompt(self, messages):
        # Render messages into a formatted prompt using the loaded Jinja template
        return self.template.render(
            bos_token=self.tokenizer.bos_token,
            messages=messages,
            add_generation_prompt=False,
        )

    def inference(self, messages):
        # Format messages into a prompt using Jinja
        prompt = self.format_prompt(messages)

        # Tokenize prompt and move input_ids to the SFT device
        input_ids = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            max_length=self.max_length,
            truncation=True,
        ).input_ids.to(self.sft_device)
        prompt_length = input_ids.size(1)  # Get the length of the prompt

        top_sample = None
        top_score = self.rejection_threshold

        for _ in range(self.max_samples):
            # Generate sample from SFT model, keeping only generated tokens
            outputs = self.sft_model.generate(
                input_ids,
                max_new_tokens=200,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Exclude the prompt from the decoded output to get only the generated portion
            generated_tokens = outputs[0, prompt_length:]  # Slice to keep only new tokens
            sample = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # Score sample using reward model
            score = self.evaluate_reward(sample)

            # Update top sample if the score is the highest so far
            if score >= top_score:
                top_sample = sample
                top_score = score

        return top_sample if top_sample is not None else "No valid samples found."

    def evaluate_reward(self, sample_text):
        # Tokenize sample for reward model evaluation and move inputs to the reward device
        inputs = self.tokenizer(sample_text, return_tensors="pt").to(self.reward_device)
        _, _, outputs = self.reward_model(**inputs, return_dict=True)

        # Reward scoring based on EOS token
        attention_masks = inputs['attention_mask']
        last_indices = (attention_masks.sum(dim=1) - 1).long()
        eos_value = outputs[torch.arange(outputs.size(0)), last_indices]

        return eos_value.item()
