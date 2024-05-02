import json
from collections import OrderedDict


def parse_conversation(episode):
    """Extract and parse conversation and goals from the episode."""
    conversation = episode["social_interactions"].split("\n\n")
    goals = episode["social_goals"]
    agent1, agent2 = list(goals.keys())
    parsed_conversation = []
    for i, utterance in enumerate(conversation):
        if utterance.startswith(agent1):
            speaker = agent1
        elif utterance.startswith(agent2):
            speaker = agent2
        else:
            continue  # Skip any unparsable utterances
        parsed_conversation.append((speaker, utterance))
    return parsed_conversation, goals


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
