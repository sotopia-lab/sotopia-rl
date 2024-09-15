import os
import re
import json
from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, List, Tuple, Union, cast

from tqdm import tqdm

# TODO: Fill in REDIS OM URL in the form of `redis://:password@host:port`
os.environ["REDIS_OM_URL"] = "redis://:QzmCUD3C3RdsR@localhost:6381"

from reverse_engineering import run_reverse_by_pk_agent

with open("../../data/sotopia_pi_openai_log_key_utterance.jsonl", 'r') as f:
    data: List[Dict[str, Any]] = [json.loads(d) for d in f.readlines()]

if not os.path.exists("../../data/episode_utterances"):
    os.makedirs("../../data/episode_utterances")
    for d in tqdm(data):
        run_reverse_by_pk_agent(d['episode_id'], True, "../../data/episode_utterances")
        run_reverse_by_pk_agent(d['episode_id'], False, "../../data/episode_utterances")

utterance_pattern = r'Utterance (\d+) by ([A-Za-z ]+)'
print(len(data))
print("turning into attributed utterances")

key_utter_dict = defaultdict(list)
max_turn_dict = defaultdict(int)
episode_id_goal_score = defaultdict(float)

for d in tqdm(data):
    for uttr_key, attributed_uttr in d['key_utterance_judgement'].items():
        episode_id_goal_score[d['episode_id']] = d['goal_score']
        match = re.search(utterance_pattern, uttr_key)
        if match:
            turn_number = match.group(1)
            agent_name = match.group(2)
        else:
            raise Exception(f"Utterance key not in correct format: {uttr_key}")
        if agent_name != d['agent']:
            continue
        
        max_turn_dict[f"{d['episode_id']}-{agent_name}"] = max(max_turn_dict[f"{d['episode_id']}-{agent_name}"], int(turn_number))
        
        if attributed_uttr[1] == "NO" or attributed_uttr[1] == -1:
            continue
        
        elif attributed_uttr[1] == "YES":
            key_utter_dict[f"{d['episode_id']}-{agent_name}"].append(int(turn_number))
        
        else:
            raise Exception(f"Attribution not in correct format: {attributed_uttr[1]}")

key_utter_dict = dict(key_utter_dict)
max_turn_dict = dict(max_turn_dict)

discounting_factor = 0.9
attribution_dict = defaultdict(dict)

def get_attribution_dict(hash_key: str):
    episode_id, agent_name = hash_key.split("-")
    attribution_list = [0] * (max_turn_dict[f"{episode_id}-{agent_name}"] + 1)
    for turn in sorted(key_utter_dict[f"{episode_id}-{agent_name}"]):
        curr_reward = 1
        for i in range(turn, -1, -1):
            attribution_list[i] += curr_reward
            curr_reward *= discounting_factor
    
    # normalize the attribution
    max_attribution = max(attribution_list)
    attribution_list = [a / max_attribution for a in attribution_list]
    
    turn_reward_dict = {}
    for i in range(0, len(attribution_list)):
        turn_reward_dict[i] = attribution_list[i]
    attribution_dict[f"{episode_id}-{agent_name}"] = turn_reward_dict

for hash_key in key_utter_dict.keys():
    get_attribution_dict(hash_key)

attribution_dict = dict(attribution_dict)

print("turning into attributed utterances")
# randomly sample a few episodes to check the attribution

print(len(attribution_dict))

attributed_data = []
for hash_key in attribution_dict:
    episode_id, agent_name = hash_key.split("-")
    for turn_number in attribution_dict[hash_key]:
        utterance_path = f"../../data/episode_utterances/{episode_id}-{agent_name}-{turn_number}.json"
        if not os.path.exists(utterance_path):
            raise Exception(f"Utterance not found: {utterance_path}")
        with open(f"../../data/episode_utterances/{episode_id}-{agent_name}-{turn_number}.json", 'r') as f:
            sotopia_utterance = json.load(f)
        
        new_utterance = deepcopy(sotopia_utterance)
        new_utterance['attribution'] = attribution_dict[hash_key][turn_number]
        new_utterance['turn_number'] = turn_number
        new_utterance['goal_score'] = episode_id_goal_score[episode_id]
        
        attributed_data.append(new_utterance)

def calc_reward(utter_attrib: float, goal_score: float) -> float:
    return utter_attrib * goal_score

sotopia_pi_utterance_reward = []
for d in tqdm(attributed_data):
    sotopia_pi_utterance_reward.append(
        {
            "instruction": d['prompt'],
            "input": "",
            "output": d['result'],
            "value": calc_reward(d['attribution'], d['goal_score']),
            "system": "",
            "history": []
        }
    )

with open("../../data/sotopia_pi_reward_key_utterance.json", 'w') as f:
    json.dump(sotopia_pi_utterance_reward, f, indent=4)