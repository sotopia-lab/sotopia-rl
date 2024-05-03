import json

from tqdm import tqdm

from human_annotate.google_form_api_wrapper import get_form
from utils.preprocess import extract_goal_scores

with open("../sotopia_episodes_v1.jsonl", "r") as f:
    episodes = [json.loads(line) for line in f]

with open("../data/form_ids.txt", "r") as f:
    form_ids = f.readlines()

form_ids = [form_id.strip() for form_id in form_ids]

example_episodes = []
visited = set()
for form_id in tqdm(form_ids):
    form = get_form(form_id)
    episode_id = form["info"]["title"].split(" ")[-1]
    for episode in episodes:
        if episode["episode_id"] == episode_id and episode_id not in visited:
            visited.add(episode_id)
            example_episodes.append(episode)
            break

with open("../data/example_episodes.jsonl", "w") as f:
    for episode in example_episodes:
        f.write(json.dumps(episode) + "\n")

example_episodes_with_scores = extract_goal_scores(example_episodes)
with open("../data/example_episodes_with_scores.jsonl", "w") as f:
    for episode in example_episodes_with_scores:
        f.write(json.dumps(episode) + "\n")
