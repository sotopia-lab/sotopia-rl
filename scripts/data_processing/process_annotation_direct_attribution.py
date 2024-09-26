import json
import os
import re
from copy import deepcopy
from typing import Any, Dict, List

from tqdm import tqdm

# TODO: Fill in REDIS OM URL in the form of `redis://:password@host:port`
os.environ["REDIS_OM_URL"] = "redis://:QzmCUD3C3RdsR@localhost:6381"

from reverse_engineering import run_reverse_by_pk_agent

with open("../../data/sotopia_pi_openai_log_attribution.jsonl", 'r') as f:
    data: List[Dict[str, Any]] = [json.loads(d) for d in f.readlines()]

if not os.path.exists("../../data/episode_utterances"):
    os.makedirs("../../data/episode_utterances")
    for d in tqdm(data):
        run_reverse_by_pk_agent(d['episode_id'], True, "../../data/episode_utterances")
        run_reverse_by_pk_agent(d['episode_id'], False, "../../data/episode_utterances")

utterance_pattern = r'Utterance (\d+) by ([A-Za-z ]+)'

print("turning into attributed utterances")

attributed_data = []
print(len(data))
for d in tqdm(data):
    for uttr_key, attributed_uttr in d['attributed_utterances'].items():
        match = re.search(utterance_pattern, uttr_key)
        if match:
            turn_number = match.group(1)
            agent_name = match.group(2)
        else:
            raise Exception(f"Utterance key not in correct format: {uttr_key}")
        if agent_name != d['agent']:
            continue

        utterance_path = f"../../data/episode_utterances/{d['episode_id']}-{d['agent']}-{turn_number}.json"
        if not os.path.exists(utterance_path):
            raise Exception(f"Utterance not found: {utterance_path}")
        with open(f"../../data/episode_utterances/{d['episode_id']}-{d['agent']}-{turn_number}.json", 'r') as f:
            sotopia_utterance = json.load(f)

        new_utterance = deepcopy(sotopia_utterance)
        new_utterance['attribution'] = attributed_uttr[1]
        new_utterance['turn_number'] = turn_number
        new_utterance['goal_score'] = d['goal_score']

        attributed_data.append(new_utterance)


def calc_reward(utter_attrib: float, goal_score: float) -> float:
    if utter_attrib == -1:
        reward = -1.0
    else:
        reward = utter_attrib / 3 * goal_score
    return reward

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

with open("../../data/sotopia_pi_reward_direct_prompt.json", 'w') as f:
    json.dump(sotopia_pi_utterance_reward, f, indent=4)

sotopia_pi_utterance_ppo = []
for d in tqdm(attributed_data):
    sotopia_pi_utterance_ppo.append(
        {
            "instruction": d['prompt'],
            "input": "",
            "output": d["result"],
        }
    )

with open("../../data/sotopia_pi_utterance_ppo.json", 'w') as f:
    json.dump(sotopia_pi_utterance_ppo, f, indent=4)
