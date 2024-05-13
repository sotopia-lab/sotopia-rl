import os
import json
from collections import OrderedDict
from typing import Any, Dict, List, Tuple


def parse_conversation(
    episode: Dict[str, Any]
) -> Tuple[List[Tuple[str, str]], Dict[str, Dict[str, Any]]]:
    """Extract and parse conversation and goals from the episode."""
    conversation = episode["social_interactions"].split("\n\n")
    goals = episode["social_goals"]
    agent1, agent2 = list(goals.keys())
    parsed_conversation = []
    for utterance in conversation:
        if utterance.startswith(agent1):
            speaker = agent1
        elif utterance.startswith(agent2):
            speaker = agent2
        else:
            continue  # Skip any unparsable utterances
        parsed_conversation.append(
            (speaker, utterance[len(speaker) + 1 :].strip())
        )  # Strip the speaker from the utterance
    return parsed_conversation, goals


def extract_goal_scores(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    new_data = []
    for episode in data:
        scores = {}
        for i in range(2):
            agent = list(episode["agents_background"].keys())[i]
            scores[agent] = episode["rewards"][i]["goal"]
        new_episode = {**episode, "scores": scores}
        new_data.append(new_episode)
    return new_data


def add_score(data_dir: str) -> None:
    with open(os.path.join(data_dir, "example_episodes.jsonl"), "r") as f:
        data = [json.loads(line, object_pairs_hook=OrderedDict) for line in f]

    new_data = []
    for episode in data:
        scores = {}
        for i in range(2):
            agent = list(episode["agents_background"].keys())[i]
            scores[agent] = episode["rewards"][i]["goal"]
        new_episode = {**episode, "scores": scores}
        new_data.append(new_episode)

    with open(os.path.join(data_dir, "example_episodes_with_scores.jsonl"), "w") as f:
        for episode in new_data:
            f.write(json.dumps(episode) + "\n")
