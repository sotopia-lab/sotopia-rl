import os
import json
from typing import Dict, Any

def select_qualifying_episodes(episodes: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    num_turns = []
    len_episodes = []
    for episode in episodes:
        num_turns.append(episode["social_interactions"].count("said:"))
        len_episodes.append(len(episode["social_interactions"]))

    qualifying_episodes = []
    for episode in episodes:
        if (
            (
                episode["rewards"][0]["goal"] >= 8
                or episode["rewards"][0]["goal"] <= 2
            )
            and (
                episode["rewards"][1]["goal"] >= 8
                or episode["rewards"][1]["goal"] <= 2
            )
            and (episode["social_interactions"].count("said:") > 2)
            and (
                episode["experiment_model_name_pairs"][1] == "gpt-4"
                or episode["experiment_model_name_pairs"][1] == "gpt-3.5-turbo"
            )
            and (
                episode["experiment_model_name_pairs"][2] == "gpt-4"
                or episode["experiment_model_name_pairs"][2] == "gpt-3.5-turbo"
            )
        ):
            qualifying_episodes.append(episode)
    return qualifying_episodes

def create_non_repeating_sample_episodes(qualifying_episodes: list[Dict[str, Any]], num_episodes: int=30) -> list[Dict[str, Any]]:
    example_episodes = []
    visited_codename = set()
    for episode in qualifying_episodes:
        if episode["codename"] in visited_codename:
            continue
        
        example_episodes.append(episode)
        visited_codename.add(episode["codename"])
        
        if len(example_episodes) == num_episodes:
            break
    return example_episodes

def sample_episodes(data_dir: str, num_episodes: int=30) -> None:
    with open(os.path.join(data_dir, "sotopia_episodes_v1.jsonl"), "r") as f:
        episodes = [json.loads(line) for line in f]
    
    qualifying_episodes = select_qualifying_episodes(episodes)
    example_episodes = create_non_repeating_sample_episodes(qualifying_episodes, num_episodes=num_episodes)
    
    with open(os.path.join(data_dir, "example_episodes.jsonl"), "w") as f:
        for episode in example_episodes:
            f.write(json.dumps(episode) + "\n")