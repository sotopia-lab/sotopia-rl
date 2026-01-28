"""
Script 2: Score the generated outputs using Qwen-2.5-7B reward model with LoRA adapter

Note: vLLM does not support sequence classification models, so we use HuggingFace transformers
for the reward model scoring with batched inference for throughput.

Supports data parallelism via accelerate for multi-GPU scoring.
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import torch
from accelerate import Accelerator, PartialState
from accelerate.utils import gather_object
from peft import PeftModelForSequenceClassification
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score DPO pairs using reward model with LoRA"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model repo or path"
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="saves/rm_goal_0503_w_relationship_knowledge_0507/checkpoint-6800",
        help="Path to LoRA adapter checkpoint"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="data/dpo_pairs_generated.json",
        help="Path to generated pairs JSON file (output from generate_dpo_pairs.py)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/dpo_pairs_scored.json",
        help="Path to output JSON file with scores"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=4096,
        help="Maximum sequence length for tokenization"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for scoring per device (adjust based on GPU memory)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for data preprocessing"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: only process one batch"
    )
    return parser.parse_args()


def load_model_and_tokenizer(model_name, adapter_path, accelerator):
    """Load model with proper device placement for distributed inference."""
    if accelerator.is_main_process:
        print(f"Loading base model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # For batch inference with decoder-only models
    
    # For distributed inference, load model on specific device
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        num_labels=1,  # For regression task (reward scoring)
        pad_token_id=tokenizer.pad_token_id,
        trust_remote_code=True,
    )
    
    # Load LoRA adapter
    adapter_file = os.path.join(adapter_path, 'adapter_model.safetensors')
    adapter_file_bin = os.path.join(adapter_path, 'adapter_model.bin')
    
    if os.path.exists(adapter_file) or os.path.exists(adapter_file_bin):
        if accelerator.is_main_process:
            print(f"Loading LoRA adapter from: {adapter_path}")
        model = PeftModelForSequenceClassification.from_pretrained(base_model, adapter_path)
    else:
        if accelerator.is_main_process:
            print(f"Warning: No adapter found at {adapter_path}, using base model")
        model = base_model
    
    # Move model to accelerator device
    model = model.to(accelerator.device)
    model.eval()
    return model, tokenizer


def format_prompt(tokenizer, input_text, output_text):
    """Format a single input-output pair as a chat prompt."""
    messages = [
        {"role": "user", "content": input_text},
        {"role": "assistant", "content": output_text},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def score_batch(model, tokenizer, prompts, max_length, device):
    """Score a batch of prompts."""
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    )
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get reward scores from logits
    rewards = outputs.logits.squeeze(-1).cpu().tolist()
    # Handle single item case
    if isinstance(rewards, float):
        rewards = [rewards]
    return rewards


def preprocess_pair(tokenizer, pair):
    """Preprocess a single pair to create formatted prompts."""
    input_text = pair['input']
    output1 = pair['output1']
    output2 = pair['output2']
    
    prompt1 = format_prompt(tokenizer, input_text, output1)
    prompt2 = format_prompt(tokenizer, input_text, output2)
    
    return prompt1, prompt2


def main():
    args = parse_args()
    
    # Initialize accelerator for distributed inference
    accelerator = Accelerator()
    
    is_main = accelerator.is_main_process
    num_processes = accelerator.num_processes
    process_index = accelerator.process_index
    
    if is_main:
        print(f"Running with {num_processes} processes")
    
    # Load model and tokenizer with LoRA adapter
    model, tokenizer = load_model_and_tokenizer(args.model, args.adapter_path, accelerator)
    
    # Load generated pairs (all processes need to know total size)
    if is_main:
        print(f"Loading generated pairs from: {args.input_path}")
    with open(args.input_path, 'r') as f:
        pairs_data = json.load(f)
    
    total_pairs = len(pairs_data)
    if is_main:
        print(f"Total pairs to score: {total_pairs}")
    
    # Preprocess all prompts (only on main process, then broadcast indices)
    if is_main:
        print("Preprocessing prompts...")
        preprocess_fn = partial(preprocess_pair, tokenizer)
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            all_prompt_pairs = list(tqdm(
                executor.map(preprocess_fn, pairs_data),
                total=total_pairs,
                desc="Preprocessing"
            ))
        
        # Flatten prompts for batched scoring (interleave prompt1 and prompt2)
        all_prompts = []
        for prompt1, prompt2 in all_prompt_pairs:
            all_prompts.append(prompt1)
            all_prompts.append(prompt2)
    else:
        # Other processes do preprocessing too (needed for their shards)
        preprocess_fn = partial(preprocess_pair, tokenizer)
        all_prompt_pairs = [preprocess_fn(pair) for pair in pairs_data]
        all_prompts = []
        for prompt1, prompt2 in all_prompt_pairs:
            all_prompts.append(prompt1)
            all_prompts.append(prompt2)
    
    # Apply test mode limit
    if args.test:
        num_test_pairs = args.batch_size // 2
        pairs_data = pairs_data[:num_test_pairs]
        all_prompts = all_prompts[:args.batch_size]
        total_pairs = num_test_pairs
        if is_main:
            print(f"[TEST MODE] Only processing {num_test_pairs} pairs")
    
    # Shard prompts across processes for data parallelism
    total_prompts = len(all_prompts)
    prompts_per_process = (total_prompts + num_processes - 1) // num_processes
    start_idx = process_index * prompts_per_process
    end_idx = min(start_idx + prompts_per_process, total_prompts)
    
    local_prompts = all_prompts[start_idx:end_idx]
    
    if is_main:
        print(f"Scoring prompts (distributed across {num_processes} GPUs)...")
    
    # Score local shard in batches
    local_scores = []
    num_local_batches = (len(local_prompts) + args.batch_size - 1) // args.batch_size
    
    for batch_idx in tqdm(range(num_local_batches), desc=f"GPU {process_index}", disable=not is_main):
        batch_start = batch_idx * args.batch_size
        batch_end = min(batch_start + args.batch_size, len(local_prompts))
        batch_prompts = local_prompts[batch_start:batch_end]
        
        batch_scores = score_batch(model, tokenizer, batch_prompts, args.max_length, accelerator.device)
        local_scores.extend(batch_scores)
    
    # Pad local scores to same length for gathering
    local_scores_with_indices = [(start_idx + i, score) for i, score in enumerate(local_scores)]
    
    # Gather all scores from all processes
    accelerator.wait_for_everyone()
    all_scores_gathered = gather_object(local_scores_with_indices)
    
    # Only main process handles final results
    if is_main:
        # Sort by original index and extract scores
        all_scores_gathered.sort(key=lambda x: x[0])
        all_scores = [score for _, score in all_scores_gathered]
        
        # Organize results
        print("Organizing results...")
        results = []
        for i, pair in enumerate(pairs_data):
            score1 = all_scores[i * 2]
            score2 = all_scores[i * 2 + 1]
            
            input_text = pair['input']
            output1 = pair['output1']
            output2 = pair['output2']
            
            # Determine chosen and rejected based on scores
            if score1 >= score2:
                chosen = output1
                rejected = output2
                chosen_score = score1
                rejected_score = score2
            else:
                chosen = output2
                rejected = output1
                chosen_score = score2
                rejected_score = score1
            
            result = {
                "input": input_text,
                "output1": output1,
                "output2": output2,
                "score1": score1,
                "score2": score2,
                "chosen": chosen,
                "rejected": rejected,
                "chosen_score": chosen_score,
                "rejected_score": rejected_score,
                "original_output": pair.get('original_output', None),
            }
            results.append(result)
        
        # Save final results
        print(f"\nSaving results to: {args.output_path}")
        with open(args.output_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Print summary statistics
        score_diffs = [r['chosen_score'] - r['rejected_score'] for r in results]
        avg_diff = sum(score_diffs) / len(score_diffs) if score_diffs else 0
        avg_score = sum(r['score1'] + r['score2'] for r in results) / (2 * len(results)) if results else 0
        
        print(f"\n=== Summary ===")
        print(f"Total pairs scored: {len(results)}")
        print(f"Average score: {avg_score:.4f}")
        print(f"Average score difference (chosen - rejected): {avg_diff:.4f}")
        print(f"Done!")


if __name__ == "__main__":
    main()
