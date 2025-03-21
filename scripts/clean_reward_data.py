import json

with open('../data/sotopia_pi_bc_episodes_reward_goal_progress_gpt-4o.json', 'r') as f:
    data = json.load(f)

cleaned_data = []
for i, example in enumerate(data):
    if example['value'] == 0:
        continue

    cleaned_data.append(example)

with open('../data/sotopia_pi_bc_episodes_reward_goal_progress_gpt-4o_cleaned.json', 'w') as f:
    json.dump(cleaned_data, f, indent=2)