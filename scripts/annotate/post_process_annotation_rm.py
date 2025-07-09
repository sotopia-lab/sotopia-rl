import json
import os
import re
from copy import deepcopy
from typing import Any, Dict, List

import click
from db_free_reverse_engineering import run_reverse_by_pk_agent
from tqdm import tqdm


def get_attributed_data(data: List[Dict[str, Any]], utterance_pattern: str, do_normalize: bool) -> List[Dict[str, Any]]:
    attributed_data = []
    print(f"Processing {len(data)} episodes")
    error_count = 0
    for d in tqdm(data):
        total_attributed_reward = 0.0
        all_utterances = []
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
            # breakpoint()
            try:
                new_utterance['attributed_reward'] = attributed_uttr[1]["reward"]
                # new_utterance['dimension'] = attributed_uttr[1]["dimension"]
                # new_utterance['scale'] = attributed_uttr[1]["scale"]
                # new_utterance['dim_score'] = attributed_uttr[1]["dim_score"]
                new_utterance['dimension'] = "goal"
                new_utterance['scale'] = "3"
                new_utterance['dim_score'] = 10
            except:
                error_count += 1
                continue
            new_utterance['turn_number'] = turn_number
            new_utterance['goal_score'] = d['goal_score']
            total_attributed_reward += new_utterance['attributed_reward']
            all_utterances.append(new_utterance)

        if do_normalize:
            for utterance in all_utterances:
                utterance['attributed_reward'] = utterance['attributed_reward'] / total_attributed_reward * d['goal_score']

        for utterance in all_utterances:
            attributed_data.append(utterance)

    print(f"Error Count: {error_count}")
    return attributed_data


@click.command()
@click.option("--data_dir", type=str, required=True, help="Directory containing data files.")
@click.option("--input_file", type=str, required=True, help="Path to the raw JSONLINES file.")
@click.option("--reward_output_file", type=str, required=True, help="Path to the processed JSON file.")
@click.option("--do_normalize", type=bool, default=False, help="Whether to normalize the rewards.")
def main(data_dir: str, input_file: str, reward_output_file: str, do_normalize: bool = False) -> None:
    with open(os.path.join(data_dir, input_file), 'r') as f:
        data: List[Dict[str, Any]] = [json.loads(d) for d in f.readlines()]

    cache_dir = os.path.join(data_dir, "episode_utterances")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        for d in tqdm(data):
            run_reverse_by_pk_agent(d['episode_id'], True, cache_dir)
            run_reverse_by_pk_agent(d['episode_id'], False, cache_dir)

    utterance_pattern = r'Utterance (\d+) by ([A-Za-z ]+)'
    print("turning into attributed utterances")

    attributed_data = get_attributed_data(data, utterance_pattern, do_normalize=do_normalize)
    sotopia_pi_utterance_reward = []
    # breakpoint()
    for d in tqdm(attributed_data):
        sotopia_pi_utterance_reward.append(
            {
                "input": d['prompt'],
                "output": d['result'],
                "value": d['attributed_reward'],
            }
        )

    with open(os.path.join(data_dir, reward_output_file), 'w') as f:
        json.dump(sotopia_pi_utterance_reward, f, indent=4)

if __name__ == "__main__":
    main()
