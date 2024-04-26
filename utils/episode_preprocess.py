import json
from collections import OrderedDict

with open("example_episodes.jsonl", "r") as f:
    data = f.readlines()
    data = [json.loads(line, object_hook=OrderedDict) for line in data]

new_data = []
for episode in data:
    scores = {}
    for i in range(2):
        agent = list(episode["agents_background"].keys())[i]
        scores[agent] = episode["rewards"][i]["goal"]
    new_episode = {**episode, "scores": scores}
    new_data.append(new_episode)

with open("example_episodes_with_scores.jsonl", "w") as f:
    for episode in new_data:
        f.write(json.dumps(episode) + "\n")
