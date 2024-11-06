import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead


class RejectionSampler:
    def __init__(self, sft_model_path, rm_model_path, model_name, rejection_threshold=0.5, max_samples=5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rejection_threshold = rejection_threshold
        self.max_samples = max_samples

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load SFT model
        self.sft_model = AutoModelForCausalLM.from_pretrained(sft_model_path).to(self.device)

        # Load reward model
        rm_model = AutoModelForCausalLMWithValueHead.from_pretrained(rm_model_path)
        #self.reward_model = get_peft_model(rm_model).to(self.device)
        self.reward_model = rm_model.to(self.device)

    def sample_and_filter(self, prompt):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        valid_samples = []

        for _ in range(self.max_samples):
            # Generate sample from SFT model
            outputs = self.sft_model.generate(input_ids, max_length=50, do_sample=True)
            sample = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Score sample using reward model
            score = self.evaluate_reward(sample)

            # Accept sample if score is above threshold
            if score >= self.rejection_threshold:
                valid_samples.append((sample, score))

        return valid_samples

    def evaluate_reward(self, sample_text):
        # Tokenize sample for reward model evaluation
        inputs = self.tokenizer(sample_text, return_tensors="pt").to(self.device)
        _, _, outputs = self.reward_model(**inputs, return_dict=True)

        # Reward scoring based on EOS token
        attention_masks = inputs['attention_mask']
        last_indices = (attention_masks.sum(dim=1) - 1).long()
        eos_value = outputs[torch.arange(outputs.size(0)), last_indices]

        return eos_value.item()

    def inference(self, prompt):
        samples = self.sample_and_filter(prompt)
        if samples:
            print("Accepted Samples and Scores:")
            for sample, score in samples:
                print(f"Sample: {sample} | Score: {score}")
        else:
            print("No samples met the threshold.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run rejection sampling with trained SFT and RM models.")
    parser.add_argument("--sft_model_path", type=str, required=True, help="Path to the fine-tuned SFT model")
    parser.add_argument("--rm_model_path", type=str, required=True, help="Path to the reward model")
    parser.add_argument("--model_name", type=str, required=True, help="Path to the tokenizer")
    parser.add_argument("--rejection_threshold", type=float, default=0.5, help="Threshold for accepting samples")
    parser.add_argument("--max_samples", type=int, default=5, help="Number of samples to generate per prompt")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt for generating samples")

    args = parser.parse_args()
    sampler = RejectionSampler(args.sft_model_path, args.rm_model_path, args.model_name, args.rejection_threshold, args.max_samples)
    sampler.inference(args.prompt)
