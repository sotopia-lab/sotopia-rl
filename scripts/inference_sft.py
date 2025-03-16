import argparse
import json
import os
import difflib

import torch
from jinja2 import Environment, FileSystemLoader
from peft import PeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Test a model with a template and example data")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model or HF model name")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to PEFT adapter (for QLoRA)")
    parser.add_argument("--template_path", type=str, required=True, help="Path to Jinja template file")
    parser.add_argument("--example_path", type=str, required=True, help="Path to example data JSON")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum output length")
    parser.add_argument("--use_qlora", action="store_true", help="Whether to use QLoRA (automatically enables 4-bit)")
    return parser.parse_args()

def load_model_and_tokenizer(args):
    print(f"Loading model: {args.model_path}")

    # Set up tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token

    # With QLoRA, we automatically use 4-bit quantization as per the training logic
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

    # Load PEFT adapter if specified
    if args.adapter_path:
        print(f"Loading adapter from: {args.adapter_path}")
        # Check for adapter files
        adapter_path = args.adapter_path
        if os.path.exists(os.path.join(adapter_path, 'adapter_model.safetensors')) or \
           os.path.exists(os.path.join(adapter_path, 'adapter_model.bin')):
            model = PeftModelForCausalLM.from_pretrained(base_model, adapter_path)
        else:
            print(f"No adapter found at {adapter_path}, using base model")
            model = base_model
    else:
        model = base_model

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

def generate_response(model, tokenizer, prompt, max_length=512):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
        )

    response = tokenizer.decode(output[0], skip_special_tokens=False)
    return response

def extract_generated_content(full_response, prompt):
    """Extract only the newly generated content by removing the prompt prefix."""
    if prompt in full_response:
        return full_response[len(prompt):].strip()
    
    return full_response

def compare_with_groundtruth(generated, groundtruth):
    """Compare generated text with groundtruth and show differences."""
    diff = difflib.unified_diff(
        generated.splitlines(),
        groundtruth.splitlines(),
        lineterm='',
        fromfile='Generated',
        tofile='Expected'
    )
    return '\n'.join(diff)

def main():
    args = parse_args()

    model, tokenizer = load_model_and_tokenizer(args)

    with open(args.example_path, 'r') as f:
        example_data = json.load(f)

    template = load_template(args.template_path)
    
    for i, example in enumerate(example_data):
        print(f"\n===== EXAMPLE {i+1}/{len(example_data)} =====")
        
        rendered_prompt = template.render(
            messages=[{"role": "user", "content": example["input"]}],
            add_generation_prompt=True, # important for inference
        )

        full_response = generate_response(model, tokenizer, rendered_prompt, args.max_length)
        
        generated_content = extract_generated_content(full_response, rendered_prompt)

        print("\nMODEL OUTPUT:")
        print(generated_content)

        print("\nGROUNDTRUTH:")
        print(example['output'])


if __name__ == "__main__":
    main()