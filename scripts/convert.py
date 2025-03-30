import torch
from transformers import AutoModelForSequenceClassification
from trl import AutoModelForCausalLMWithValueHead
from peft import PeftConfig, get_peft_model

# 1. Fix the state_dict method in AutoModelForCausalLMWithValueHead
# This is the key fix to prevent "OrderedDict mutated during iteration"
def safe_state_dict(self, *args, **kwargs):
    """Fixed state_dict implementation that avoids OrderedDict mutation"""
    # Get the base model's state dict
    sd = self.pretrained_model.state_dict()
    
    # Safely add the v_head state dict
    v_head_sd = self.v_head.state_dict()
    for k, v in list(v_head_sd.items()):  # Use list() to create a copy for iteration
        sd[f"v_head.{k}"] = v
        
    return sd

# Apply the patch
AutoModelForCausalLMWithValueHead.state_dict = safe_state_dict

# 2. Load your classification model
classification_model = AutoModelForSequenceClassification.from_pretrained(
    "/data/haofeiy2/sotopia-rl/rm_reward_direct_default_no_goal_gpt-4o_without_goal_leak/checkpoint-4000",
    num_labels=1
)

# 3. Get the PEFT config 
peft_config_path = "/data/haofeiy2/sotopia-rl/rm_reward_direct_default_no_goal_gpt-4o_without_goal_leak/checkpoint-4000"
peft_config = PeftConfig.from_pretrained(peft_config_path)

# 4. Load the base model with value head
base_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    "/mnt/data_from_server1/models/Qwen2.5-7B-Instruct",
    num_labels=1,
    return_dict=True
)

# 5. Apply PEFT config to create a PEFT version
peft_model = get_peft_model(base_model, peft_config)

# 6. Extract adapter weights from the classification model
adapter_weights = {}
for name, param in classification_model.named_parameters():
    if "lora" in name:
        adapter_weights[name] = param.data.clone()

# 7. Now we need to match the keys correctly
print("Available keys in PEFT model:")
peft_keys = [name for name, _ in peft_model.named_parameters() if "lora" in name]
for key in peft_keys[:10]:  # Print first 10 keys for debugging
    print(f"  {key}")

print("\nAvailable keys in classification model:")
class_keys = [name for name, _ in classification_model.named_parameters() if "lora" in name]
for key in class_keys[:10]:  # Print first 10 keys for debugging
    print(f"  {key}")

# 8. Load weights one by one to avoid state_dict issues
for peft_name, param in peft_model.named_parameters():
    if "lora" in peft_name:
        # Try to find a matching key in the classification model
        matching_key = None
        for class_name in class_keys:
            if class_name.split(".")[-1] == peft_name.split(".")[-1]:  # Match the parameter name part
                matching_key = class_name
                break
        
        if matching_key and matching_key in adapter_weights:
            # Check shapes
            if param.shape == adapter_weights[matching_key].shape:
                # Directly assign the parameter data
                param.data.copy_(adapter_weights[matching_key])
                print(f"Transferred: {matching_key} → {peft_name}")
            else:
                print(f"Shape mismatch: {matching_key} {adapter_weights[matching_key].shape} vs {peft_name} {param.shape}")
        else:
            print(f"No matching key found for {peft_name}")

# 9. Save the model with adapter
peft_model.save_pretrained("/data/haofeiy2/sotopia-rl/rm_direct_default_with_valuehead")