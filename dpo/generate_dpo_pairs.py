"""
Script 1: Generate two outputs from Qwen-2.5-7B for each input using vLLM
"""
import argparse
import json

from tqdm import tqdm
from vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate two outputs per input using Qwen-2.5-7B with vLLM"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model repo or path"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="data/sotopia_grpo.json",
        help="Path to input JSON file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/dpo_pairs_generated.json",
        help="Path to output JSON file"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to process (None for all)"
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism"
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization (0.0 to 1.0)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Number of samples per batch (for progress tracking and memory management)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: only process one batch"
    )
    return parser.parse_args()


def process_batch(llm, batch_data, batch_start_idx, sampling_params):
    """Process a batch of inputs and return results."""
    # Prepare prompts for this batch (2 per input for output1 and output2)
    all_prompts = []
    prompt_to_idx = []
    
    for local_idx, example in enumerate(batch_data):
        input_text = example['input']
        messages = [{"role": "user", "content": input_text}]
        # Add two prompts for each input
        all_prompts.append(messages)
        prompt_to_idx.append((local_idx, 1))
        all_prompts.append(messages)
        prompt_to_idx.append((local_idx, 2))
    
    # Generate all outputs in batch
    outputs = llm.chat(
        messages=all_prompts,
        sampling_params=sampling_params,
    )
    
    # Organize results
    results = [{
        "input": example['input'],
        "output1": None,
        "output2": None,
        "original_output": example.get('output', None),
    } for example in batch_data]
    
    for output, (local_idx, output_num) in zip(outputs, prompt_to_idx):
        generated_text = output.outputs[0].text.strip()
        if output_num == 1:
            results[local_idx]["output1"] = generated_text
        else:
            results[local_idx]["output2"] = generated_text
    
    return results


def main():
    args = parse_args()
    
    # Load input data
    print(f"Loading input data from: {args.input_path}")
    with open(args.input_path, 'r') as f:
        input_data = json.load(f)
    
    if args.num_samples is not None:
        input_data = input_data[:args.num_samples]
    
    total_samples = len(input_data)
    print(f"Processing {total_samples} samples in batches of {args.batch_size}...")
    
    # Initialize vLLM
    print(f"Loading model: {args.model}")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )
    
    # Sampling parameters
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    
    # Process in batches for better progress tracking and memory management
    all_results = []
    num_batches = (total_samples + args.batch_size - 1) // args.batch_size
    
    if args.test:
        num_batches = 1
        print("[TEST MODE] Only processing 1 batch")
    
    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, total_samples)
        batch_data = input_data[start_idx:end_idx]
        
        batch_results = process_batch(llm, batch_data, start_idx, sampling_params)
        all_results.extend(batch_results)
        
        # Save intermediate results after each batch
        print(f"\nSaving intermediate results ({end_idx}/{total_samples} samples)...")
        with open(args.output_path, 'w') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nDone! Generated {len(all_results)} pairs.")
    print(f"Results saved to: {args.output_path}")


if __name__ == "__main__":
    main()
