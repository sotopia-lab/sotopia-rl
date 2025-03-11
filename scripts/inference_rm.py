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
        description="Run inference with a trained SotopiaRM model on a single input"
    )
    # Model parameters
    parser.add_argument("--base_model", type=str, default="mistralai/Mistral-7B-v0.1",
                        help="Base model name or path")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to saved checkpoint directory")
    parser.add_argument("--template_string", type=str, default=None,
                        help="Jinja2 template as a string (optional)")
    parser.add_argument("--template_path", type=str, default=None,
                        help="Path to Jinja2 template file (optional)")
    parser.add_argument("--prompt_file", type=str, default=None,
                        help="Path to JSON file containing conversation data (optional)")
    parser.add_argument("--use_4bit", action="store_true",
                        help="Whether to use 4-bit quantization for inference")

    return parser.parse_args()

def load_model_and_tokenizer(args):
    print(f"Loading base model: {args.base_model}")

    # Set up tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token

    # Set up model with potential quantization
    if args.use_4bit:
        print("Using 4-bit quantization for inference")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=quantization_config
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    # Load PEFT adapter
    adapter_path = os.path.join(args.checkpoint_path, 'adapter_model')
    if os.path.exists(adapter_path + '.safetensors') or os.path.exists(adapter_path + '.bin'):
        print(f"Loading adapter from: {args.checkpoint_path}")
        peft_model = PeftModelForCausalLM.from_pretrained(base_model, args.checkpoint_path)
    else:
        print(f"No adapter found at {adapter_path}, using base model")
        peft_model = base_model

    # Convert to value head model
    model = AutoModelForCausalLMWithValueHead.from_pretrained(peft_model)

    # Load value head weights
    value_head_path = os.path.join(args.checkpoint_path, 'value_head.pt')
    if os.path.exists(value_head_path):
        print(f"Loading value head from: {value_head_path}")
        value_head_state_dict = torch.load(value_head_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
        model.v_head.load_state_dict(value_head_state_dict)
    else:
        print(f"WARNING: No value head weights found at {value_head_path}")

    # Ensure model is in evaluation mode
    model.eval()

    return model, tokenizer

def load_template_from_file(template_path):
    # Get directory and filename
    template_dir = os.path.dirname(template_path)
    template_file = os.path.basename(template_path)

    # If template is in the current directory, adjust accordingly
    if not template_dir:
        template_dir = "."

    env = Environment(loader=FileSystemLoader(template_dir))
    # Add the "tojson" filter
    env.filters['tojson'] = lambda obj: json.dumps(obj)
    return env.get_template(template_file)

def create_template_from_string(template_string):
    env = Environment()
    env.filters['tojson'] = lambda obj: json.dumps(obj)
    return env.from_string(template_string)

def evaluate_single_prompt(model, tokenizer, prompt):
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

    # Load conversation data or use default.
    # Here we assume that the conversation item includes "instruction" and "output"
    if args.prompt_file:
        with open(args.prompt_file, 'r') as f:
            item = json.load(f)
            print(f"Loaded data from file: {args.prompt_file}")
    else:
        item = {
            "instruction": (
                "Imagine you are Sasha Ramirez, your task is to act/speak as Sasha Ramirez would, "
                "keeping in mind Sasha Ramirez's social goal. Here is the context of the interaction: "
                "Scenario: Agent1 gets extremely upset and loud while playing video games, causing Agent2 "
                "to experience anxiety attacks. [Further context here]"
            ),
            "output": "Your expected assistant response (if any) can be provided here."
        }
        print("Using default conversation data")

    # Load template from file, string, or use default template.
    if args.template_path:
        template = load_template_from_file(args.template_path)
        print(f"Loaded template from file: {args.template_path}")
    elif args.template_string:
        template = create_template_from_string(args.template_string)
        print("Using template from string input")
    else:
        print("No template specified. Using default Jinja template.")
        # Default template supports bos_token, messages, and add_generation_prompt.
        default_template = (
            "{{ bos_token }}\n"
            "{% for message in messages %}"
            "{% if message.role == 'user' %}User: {{ message.content }}\n"
            "{% elif message.role == 'assistant' %}Assistant: {{ message.content }}\n"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}\nGeneration prompt added.\n{% endif %}"
        )
        template = create_template_from_string(default_template)

    import pdb; pdb.set_trace()
    # Render the prompt using the new message structure.
    rendered_text = template.render(
        bos_token=tokenizer.bos_token,
        messages=[
            {"role": "user", "content": item["instruction"], "tool_calls": None},
            {"role": "assistant", "content": item["output"], "tool_calls": None}
        ],
        add_generation_prompt=False
    )
    print("\n===== FORMATTED PROMPT =====")
    print(rendered_text)
    print("===========================\n")

    # Evaluate the prompt to obtain a reward score
    reward = evaluate_single_prompt(model, tokenizer, rendered_text)

    print("\n===== REWARD SCORE =====")
    print(f"Score: {reward:.6f}")
    print("=======================\n")

if __name__ == "__main__":
    main()
