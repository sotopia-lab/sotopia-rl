import argparse
import json
import os

import torch
from jinja2 import Environment, FileSystemLoader
from peft import PeftModelForSequenceClassification
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test a reward model with a template and example data"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to base model or HF model name",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Path to saved checkpoint directory",
    )
    parser.add_argument(
        "--template_path", type=str, required=True, help="Path to Jinja template file"
    )
    parser.add_argument(
        "--example_path", type=str, required=True, help="Path to example data JSON"
    )
    return parser.parse_args()


def load_model_and_tokenizer(args):
    print(f"Loading base model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    print("Using full precision model")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.adapter_path,
        device_map="auto",
        num_labels=1,  # For regression task
        pad_token_id=tokenizer.pad_token_id,  # very important to add this
    )

    def print_named_parameters(model, keyword="score"):
        for name, param in model.named_parameters():
            if keyword in name:
                print(
                    f"{name}: mean={param.data.mean():.4f}, std={param.data.std():.4f}"
                )
            else:
                print("did not load score_weights")

    print_named_parameters(model, keyword="score")

    model.eval()

    return model, tokenizer


def load_template(template_path):
    template_dir = os.path.dirname(template_path)
    template_file = os.path.basename(template_path)

    if not template_dir:
        template_dir = "."

    env = Environment(loader=FileSystemLoader(template_dir))
    env.filters["tojson"] = lambda obj: json.dumps(obj)
    return env.get_template(template_file)


def evaluate_prompt(model, tokenizer, prompt, index=None):
    print(f"\n[DEBUG] Prompt [{index}]:")
    print(prompt)
    print("[DEBUG] Decoded Input IDs:")
    prompt = prompt.strip()
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True)
    print(tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=False))

    # Check input length
    print(f"[DEBUG] Input length: {encoded['input_ids'].shape[-1]} tokens")
    # inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    device = next(model.parameters()).device

    inputs = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Get reward score directly from the logits
    # reward = outputs.logits.squeeze().cpu().item()
    logits = outputs.logits.squeeze()
    print(f"[DEBUG] Raw logits: {logits}")
    reward = logits.cpu().item()
    return reward


def main():
    args = parse_args()

    model, tokenizer = load_model_and_tokenizer(args)

    with open(args.example_path, "r") as f:
        example_data = json.load(f)

    template = load_template(args.template_path)
    for i, example in enumerate(example_data):
        print(f"\n===== EXAMPLE {i+1}/{len(example_data)} =====")

        rendered_prompt = template.render(
            messages=[
                {"role": "user", "content": example["input"]},
                {"role": "assistant", "content": example["output"]},
            ],
            add_generation_prompt=False,
        )

        # reward = evaluate_prompt(model, tokenizer, rendered_prompt)
        reward = evaluate_prompt(model, tokenizer, rendered_prompt, index=i + 1)
        gth_reward = example.get("value")

        print(f"REWARD SCORE: {reward:.6f}")
        if gth_reward is not None:
            print(f"GTH REWARD: {gth_reward:.6f}")
        else:
            print("GTH REWARD: Not available")


if __name__ == "__main__":
    main()
