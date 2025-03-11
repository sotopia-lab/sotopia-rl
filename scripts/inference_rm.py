import argparse
import json
import os

import torch
from jinja2 import Environment, FileSystemLoader
from peft import PeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import AutoModelForCausalLMWithValueHead


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test a reward model with a template and example data"
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to base model or HF model name")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to saved checkpoint directory")
    parser.add_argument("--template_path", type=str, required=True, help="Path to Jinja template file")
    parser.add_argument("--example_path", type=str, required=True, help="Path to example data JSON")
    parser.add_argument("--use_qlora", action="store_true", help="Whether to use QLoRA (automatically enables 4-bit)")
    return parser.parse_args()

def load_model_and_tokenizer(args):
    print(f"Loading base model: {args.model_path}")

    # Set up tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Set up model with potential quantization
    if args.use_qlora:
        print("Using QLoRA with 4-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=quantization_config
        )
    else:
        print("Using full precision model")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    # Load PEFT adapter
    adapter_path = os.path.join(args.adapter_path, 'adapter_model')
    if os.path.exists(adapter_path + '.safetensors') or os.path.exists(adapter_path + '.bin'):
        print(f"Loading adapter from: {args.adapter_path}")
        peft_model = PeftModelForCausalLM.from_pretrained(base_model, args.adapter_path)
    else:
        print(f"No adapter found at {adapter_path}, using base model")
        peft_model = base_model

    # Convert to value head model
    model = AutoModelForCausalLMWithValueHead.from_pretrained(peft_model)

    # Load value head weights
    value_head_path = os.path.join(args.adapter_path, 'value_head.pt')
    if os.path.exists(value_head_path):
        print(f"Loading value head from: {value_head_path}")
        value_head_state_dict = torch.load(value_head_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
        model.v_head.load_state_dict(value_head_state_dict)
    else:
        print(f"WARNING: No value head weights found at {value_head_path}")

    # Ensure model is in evaluation mode
    model.eval()

    return model, tokenizer

def load_template(template_path):
    template_dir = os.path.dirname(template_path)
    template_file = os.path.basename(template_path)
    
    if not template_dir:
        template_dir = "."
        
    env = Environment(loader=FileSystemLoader(template_dir))
    env.filters['tojson'] = lambda obj: json.dumps(obj)
    return env.get_template(template_file)

def evaluate_prompt(model, tokenizer, prompt):
    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    # Get the device from the model
    device = next(model.parameters()).device

    # Move inputs to the device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate reward score
    with torch.no_grad():
        # Assumes the model returns logits, hidden states, and a value tensor
        _, _, values = model(**inputs)

    # Extract reward score from the last token's value
    last_index = inputs['attention_mask'].sum(dim=1) - 1
    reward = values[0, last_index].cpu().item()

    return reward

def main():
    args = parse_args()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args)
    
    # Load example data
    with open(args.example_path, 'r') as f:
        example_data = json.load(f)
    
    # Load template
    template = load_template(args.template_path)
    
    # Render the prompt
    rendered_prompt = template.render(
        messages=[
            {"role": "user", "content": example_data.get("instruction", example_data.get("input", "")), "tool_calls": None},
            {"role": "assistant", "content": example_data.get("output", ""), "tool_calls": None}
        ],
        add_generation_prompt=False
    )
    
    print("\n===== FORMATTED PROMPT =====")
    print(rendered_prompt)
    print("===========================\n")
    
    # Evaluate prompt to get reward score
    reward = evaluate_prompt(model, tokenizer, rendered_prompt)
    
    print("\n===== REWARD SCORE =====")
    print(f"Score: {reward:.6f}")
    print("=======================\n")
    
if __name__ == "__main__":
    main()