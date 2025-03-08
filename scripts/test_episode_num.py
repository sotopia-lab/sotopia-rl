import json

with open('../data/sotopia_pi_bc_episodes_reward_o3-mini.json') as f:
    dataset = json.load(f)

settings = []
for data in dataset:
    parts = data['instruction'].split('\n')
    # part[8] is scenario, part[9] is agent combo
    setting = parts[8] + parts[9]
    if setting not in settings:
        settings.append(setting) 
print('Total utterances:', len(dataset))
print('Total AgentEnvCombos:', len(settings))