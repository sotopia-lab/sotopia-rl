from typing import Dict, List


def extract_goal_scores(data: List) -> List:
    new_data = []
    for episode in data:
        scores = {}
        for i in range(2):
            agent = list(episode["agents_background"].keys())[i]
            scores[agent] = episode["rewards"][i]["goal"]
        new_episode = {**episode, "scores": scores}
        new_data.append(new_episode)
    return new_data
