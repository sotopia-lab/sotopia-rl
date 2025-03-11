import argparse
import json

import torch
from jinja2 import Environment, FileSystemLoader
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a fine-tuned SotopiaSFT model")

    # Model parameters
    parser.add_argument("--base_model", type=str, default="mistralai/Mistral-7B-v0.1",
                      help="Base model name or path")
    parser.add_argument("--lora_checkpoint", type=str, required=True,
                      help="Path to LoRA adapter checkpoint")
    parser.add_argument("--template_path", type=str, required=True,
                      help="Path to Jinja2 template file")
    parser.add_argument("--use_4bit", action="store_true",
                      help="Whether to use 4-bit quantization for inference")
    parser.add_argument("--max_length", type=int, default=2048,
                      help="Maximum sequence length")
    parser.add_argument("--temperature", type=float, default=0.7,
                      help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                      help="Top-p sampling parameter")
    parser.add_argument("--input_file", type=str, default=None,
                      help="JSON file containing input conversations (optional)")

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

    # Load LoRA adapter
    print(f"Loading LoRA checkpoint from: {args.lora_checkpoint}")
    model = PeftModel.from_pretrained(base_model, args.lora_checkpoint)

    # Ensure model is in evaluation mode
    model.eval()

    return model, tokenizer

def load_template(template_path):
    # Load and return the Jinja2 template
    env = Environment(loader=FileSystemLoader("/".join(template_path.split("/")[:-1])))
    return env.get_template(template_path.split("/")[-1])

def generate_response(model, tokenizer, template, conversation, args):
    # Render the template with the conversation
    system_content = ""
    if conversation.get("scenario"):
        system_content += f"Scenario: {conversation['scenario']}\n"
    if conversation.get("user_persona"):
        system_content += f"User Persona: {conversation['user_persona']}\n"
    if conversation.get("agent_persona"):
        system_content += f"Agent Persona: {conversation['agent_persona']}\n"
    if conversation.get("goal"):
        system_content += f"Goal: {conversation['goal']}\n"

    if not system_content:
        system_content = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

    messages = [{"role": "system", "content": system_content}]

    if "history" in conversation:
        messages.extend(conversation["history"])

    messages.append({"role": "user", "content": conversation["current_message"]})

    prompt = template.render(messages=messages)

    print("\n===== INPUT =====")
    print(prompt)
    print("=================\n")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)


    # Generate the response
    with torch.no_grad():
        output = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode the response
    generated_text = tokenizer.decode(output[0], skip_special_tokens=False)

    # Extract assistant's response
    if "<|im_start|>assistant" in generated_text and "<|im_end|>" in generated_text:
        response = generated_text.split("<|im_start|>assistant")[-1].split("<|im_end|>")[0]
        response = response.strip()
    else:
        # Fallback if special tokens aren't properly generated
        response = generated_text[len(prompt):].strip()

    print("\n===== GENERATED RESPONSE =====")
    print(response)
    print("==============================\n")

    return response

def interactive_mode(model, tokenizer, template, args):
    """Run the model in interactive mode."""
    print("\n===== INTERACTIVE MODE =====")
    print("Type your messages below. Type 'exit' to quit.")
    print("===============================\n")

    # Start with an empty conversation history
    history = []

    # Get basic scenario and persona information
    scenario = input("Enter scenario (press Enter to skip): ")
    user_persona = input("Enter user persona (press Enter to skip): ")
    agent_persona = input("Enter agent persona (press Enter to skip): ")
    goal = input("Enter conversation goal (press Enter to skip): ")

    while True:
        # Get user input
        user_message = input("\nYou: ")
        if user_message.lower() == 'exit':
            break

        # Prepare conversation object
        conversation = {
            "scenario": scenario,
            "user_persona": user_persona,
            "agent_persona": agent_persona,
            "goal": goal,
            "history": history.copy(),
            "current_message": user_message
        }

        # Generate response
        assistant_response = generate_response(model, tokenizer, template, conversation, args)
        print(f"\nAssistant: {assistant_response}")

        # Update history
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": assistant_response})

def batch_mode(model, tokenizer, template, input_file, args):
    """Run the model on a batch of inputs from a file."""
    print(f"\n===== BATCH MODE: Processing {input_file} =====")

    # Load inputs from file
    with open(input_file, 'r') as f:
        inputs = json.load(f)

    results = []

    for i, input_data in enumerate(inputs):
        print(f"\nProcessing example {i+1}/{len(inputs)}")

        conversation = input_data.get('conversation', input_data)
        response = generate_response(model, tokenizer, template, conversation, args)

        # Save result
        result = input_data.copy()
        if 'conversation' in result:
            result['conversation']['model_response'] = response
        else:
            result['model_response'] = response

        results.append(result)

    # Save results to file
    output_file = input_file.replace('.json', '_results.json')
    if output_file == input_file:
        output_file = input_file.split('.')[0] + '_results.json'

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_file}")

def main():
    args = parse_args()

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args)

    # Load template
    template = load_template(args.template_path)

    # Run in batch mode if input file provided, otherwise interactive mode
    if args.input_file:
        batch_mode(model, tokenizer, template, args.input_file, args)
    else:
        interactive_mode(model, tokenizer, template, args)

if __name__ == "__main__":
    main()
