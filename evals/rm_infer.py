import os
from typing import Tuple

from peft import PeftModel
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead


def load_model_with_adapter_and_value_head(
    model_path: str, adapter_path: str, value_head_file: str="value_head.safetensors"
) -> Tuple[AutoModelForCausalLMWithValueHead, AutoTokenizer]:
    # Load the base model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Apply the LoRA adapter
    model = PeftModel.from_pretrained(model, adapter_path)

    # Wrap with the value head model
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model)

    # Load value head weights from the safetensors file
    file_path = os.path.join(adapter_path, value_head_file)
    with safe_open(file_path, framework="pt") as f:
        tensor_dict = {key: f.get_tensor(key) for key in f.keys()}
    model.load_state_dict(tensor_dict, strict=False)

    return model, tokenizer

if __name__ == "__main__":
    adapter_path = '../saves/gemma-2-2b/lora/rm/checkpoint-2000/'
    model_path = '/workspace/gemma-2-2b-it'
    model, tokenizer = load_model_with_adapter_and_value_head(model_path, adapter_path)
